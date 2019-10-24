#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "RayGen.h"
#include "Shading.h"
#include "TriFetch.h"

#pragma warning (disable: 3078) // this doesn't seem to work with the new HLSL compiler...

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

Texture2D<float4> g_materialTextures[] : register(t100);

#if QUAD_READ_GROUPSHARED_FALLBACK
groupshared float2 tileUVs[TILE_SIZE];
#endif

struct ShadePixel
{
    uint id; // mesh + primitive IDs
    uint sampleMask;
};

#define SORT_SIZE AA_SAMPLES
#define SORT_T ShadePixel
#define SORT_CMP_LESS(a, b) (a.id < b.id)
#include "Sort.h"

struct ShadeQuad
{
    uint id; // mesh + primitive IDs
// TODO: collapse down to 8 bytes when 16x MSAA
// TODO: do I actually need to track the sample mask, or just a count? That'd be less space.
    uint sampleMask[QUAD_SIZE]; // sample mask for each pixel in the quad
};
groupshared ShadeQuad shadeQuads[MAX_SHADE_QUADS_PER_TILE];

groupshared float3 tileFramebuffer[TILE_SIZE];

float3 ShadeQuadThread(
    uint threadID, // for groupshared fallback
    float3 rayDir, uint meshID, uint primID, float3 uvw
)
{
    uint materialID = g_meshInfo[meshID].materialID;
    TriInterpolated tri = triFetchAndInterpolate(meshID, primID, uvw);

#if QUAD_READ_GROUPSHARED_FALLBACK
    // groupshared fallback
    tileUVs[threadID] = tri.uv;
    GroupMemoryBarrierWithGroupSync();
    uint threadID00 = threadID & ~(QUAD_SIZE - 1);
    float2 uv00 = tileUVs[threadID00 + 0];
    float2 uv10 = tileUVs[threadID00 + 1];
    float2 uv01 = tileUVs[threadID00 + 2];
#else
    // Doesn't work in compute shaders (it's spec'd to), unless you disable validation in the HLSL compiler
    // and enable D3D12ExperimentalShaderModels prior to device creation.
    // Response from Tex at MSFT: Known issue that will be fixed in Vibranium (Spring 2020).
    // Current GitHub master build has this fixed, but you still need a new DXIL.dll for signed shaders
    // for the whole thing to work...
    float2 uv00 = QuadReadLaneAt(tri.uv, 0);
    float2 uv10 = QuadReadLaneAt(tri.uv, 1);
    float2 uv01 = QuadReadLaneAt(tri.uv, 2);
#endif

    float2 uvDx = uv10 - uv00;
    float2 uvDy = uv01 - uv00;

    float3 diffuseColor = g_materialTextures[materialID * 2 + 0].SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb;

    float3 normal = g_materialTextures[materialID * 2 + 1].SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb * 2 - 1;
    float gloss = 128;
    AntiAliasSpecular(normal, gloss);

    float3x3 tbn = float3x3(
        normalize(tri.tangent),
        normalize(tri.bitangent),
        normalize(tri.normal));
    normal = normalize(mul(normal, tbn));

    float3 viewDir = normalize(rayDir);
    float specularMask = .1; // TODO: read the texture

    float3 outputColor = Shade(
        diffuseColor,
        shadeConstants.ambientColor,
        float3(.56f, .56f, .56f),
        specularMask,
        gloss,
        normal,
        viewDir,
        shadeConstants.sunDirection,
        shadeConstants.sunColor);

    return outputColor;
}

// this swizzling enables the use of QuadRead* lane sharing intrinsics in a compute shader
void threadIndexToQuadSwizzle(uint threadID, out uint localX, out uint localY)
{
    // address bit layout (high to low)
    // 8x8 tile: yyxxyx
    // 8x4 tile:  yxxyx
    localX = (((threadID >> 2                    ) << 1) | ( threadID       & 1)) & ((1 << TILE_DIM_LOG2_X) - 1);
    localY = (((threadID >> (1 + TILE_DIM_LOG2_X)) << 1) | ((threadID >> 1) & 1)) & ((1 << TILE_DIM_LOG2_Y) - 1);
}

[numthreads(TILE_SIZE, 1, 1)]
[RootSignature(
    "CBV(b0),"
    "CBV(b1),"
    "DescriptorTable(UAV(u2, numDescriptors = 3)),"
    "DescriptorTable(SRV(t1, numDescriptors = 3)),"
    "DescriptorTable(SRV(t100, numDescriptors = unbounded)),"
    "StaticSampler(s0, maxAnisotropy = 8),"
)]
void ShadeQuads(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileX = groupID.x;
    uint tileY = groupID.y;
    uint tileIndex = tileY * dynamicConstants.tilesX + tileX;
    uint threadID = groupThreadID.x;
    uint localX;
    uint localY;
    threadIndexToQuadSwizzle(threadID, localX, localY);
    uint pixelX = tileX * TILE_DIM_X + localX;
    uint pixelY = tileY * TILE_DIM_Y + localY;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM_X;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM_Y;

    uint tileTriCount = g_tileTriCounts[tileIndex];
    if (tileTriCount <= 0)
    {
        // no triangles overlap this tile
        g_screenOutput[uint2(pixelX, pixelY)] = float4(0, 0, 1, 1);
        return;
    }
    else if (tileTriCount > TILE_MAX_TRIS)
    {
        // tile tri list overflowed
        g_screenOutput[uint2(pixelX, pixelY)] = float4(1, 0, 0, 1);
        return;
    }

    float nearestT[AA_SAMPLES];
    uint nearestID[AA_SAMPLES];
    {for (uint s = 0; s < AA_SAMPLES; s++)
    {
        nearestT[s] = FLT_MAX;
        nearestID[s] = BAD_TRI_ID;
    }}
    for (uint tileTriIndex = 0; tileTriIndex < tileTriCount; tileTriIndex++)
    {
        uint id = g_tileTris[tileIndex].id[tileTriIndex];
        uint meshID = id >> 16;
        uint primID = id & 0xffff;

        Triangle tri = triFetch(meshID, primID);
        
        for (uint s = 0; s < AA_SAMPLES; s++)
        {
            float3 rayOrigin;
            float3 rayDir;
            GenerateCameraRay(
                uint2(pixelDimX, pixelDimY),
                float2(pixelX, pixelY) + AA_SAMPLE_OFFSET_TABLE[s],
                rayOrigin, rayDir);

            float4 uvwt = triIntersect(rayOrigin, rayDir, tri);

            if (uvwt.x >= 0 && uvwt.y >= 0 && uvwt.z >= 0 &&
                uvwt.w < nearestT[s])
            {
                nearestT[s] = uvwt.w;
                nearestID[s] = id;
            }
        }
    }

// TODO: do I actually need to track the sample mask, or just a count? That'd be less space.
    // thread-local list of triangles to shade
    ShadePixel shadePixel[AA_SAMPLES];
    uint shadePixelCount = 0;
    {for (uint s = 0; s < AA_SAMPLES; s++)
    {
        shadePixel[s].id = BAD_TRI_ID;
    }}
    {for (uint s = 0; s < AA_SAMPLES; s++)
    {
        uint id = nearestID[s];
        if (id == BAD_TRI_ID)
            continue; // no hit

        uint s2;
        for (s2 = 0; s2 < shadePixelCount; s2++)
        {
            if (shadePixel[s2].id == id)
            {
                shadePixel[s2].sampleMask |= 1 << s;
                break;
            }
        }

        // this triangle wasn't already in the list
        if (s2 == shadePixelCount)
        {
            shadePixel[shadePixelCount].id = id;
            shadePixel[shadePixelCount].sampleMask = 1 << s;
            shadePixelCount++;
        }
    }}

    // sort the compacted list
    sortBitonic(shadePixel);

    /*uint shadeQuadCount = 0;

    uint quadID = threadID / QUAD_SIZE;
    uint quadThreadIndex = threadID & (QUAD_SIZE - 1);

    for (uint n = 0; n < shadePixelCount; n++)
    {
        uint id = shadePixel[n].id;
    }*/

    // TODO: centroid
    float3 rayOriginShade;
    float3 rayDirShade;
    GenerateCameraRay(
        uint2(pixelDimX, pixelDimY),
        float2(pixelX, pixelY),
        rayOriginShade, rayDirShade);

// TODO: each quad-size group of threads will pull shade-quads off the list and accumulate into the tile
// framebuffer *without* synchronization (we are running a warp-sized tile, so it should be OK)
// This means we'll need to store the quadID in the shade-quad
    tileFramebuffer[threadID] = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint n = 0; n < shadePixelCount; n++)
    {
        uint id = shadePixel[n].id;

        if (id == BAD_TRI_ID)
            continue; // no hit

        uint meshID = id >> 16;
        uint primID = id & 0xffff;

        Triangle tri = triFetch(meshID, primID);
        float3 uvw = triIntersectNoFail(rayOriginShade, rayDirShade, tri).xyz;

        uint sampleCount = countbits(shadePixel[n].sampleMask);

// TODO: accum into the shade-quad's address, not our address
        tileFramebuffer[threadID] += sampleCount * ShadeQuadThread(
            threadID,
            rayDirShade, meshID, primID, uvw);
    }

    GroupMemoryBarrierWithGroupSync();

    g_screenOutput[uint2(pixelX, pixelY)] = float4(tileFramebuffer[threadID] / AA_SAMPLES, 1);
}

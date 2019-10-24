#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "RayGen.h"
#include "Shading.h"
#include "TriFetch.h"

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

Texture2D<float4> g_materialTextures[] : register(t100);

#if QUAD_READ_GROUPSHARED_FALLBACK
groupshared float2 tileUVs[TILE_SIZE];
#endif

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
    uint threadID00 = threadID & ~3;
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

    uint s;
    float nearestT[AA_SAMPLES];
    uint nearestID[AA_SAMPLES];
    for (s = 0; s < AA_SAMPLES; s++)
    {
        nearestT[s] = FLT_MAX;
        nearestID[s] = ~uint(0);
    }

    for (uint tileTriIndex = 0; tileTriIndex < tileTriCount; tileTriIndex++)
    {
        uint id = g_tileTris[tileIndex].id[tileTriIndex];
        uint meshID = id >> 16;
        uint primID = id & 0xffff;

        Triangle tri = triFetch(meshID, primID);
        
        for (s = 0; s < AA_SAMPLES; s++)
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

    float3 outColor = 0;
    for (s = 0; s < AA_SAMPLES; s++)
    {
        if (nearestID[s] == ~uint(0))
            continue; // no hit

        float3 rayOrigin;
        float3 rayDir;
        GenerateCameraRay(
            uint2(pixelDimX, pixelDimY),
            float2(pixelX, pixelY) + AA_SAMPLE_OFFSET_TABLE[s],
            rayOrigin, rayDir);

        uint meshID = nearestID[s] >> 16;
        uint primID = nearestID[s] & 0xffff;

        Triangle tri = triFetch(meshID, primID);
        float3 uvw = triIntersectNoFail(rayOrigin, rayDir, tri).xyz;

        outColor += ShadeQuadThread(
            threadID,
            rayDir, meshID, primID, uvw);
    }

    g_screenOutput[uint2(pixelX, pixelY)] = float4(outColor / AA_SAMPLES, 1);
}

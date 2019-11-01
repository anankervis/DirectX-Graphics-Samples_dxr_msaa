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

struct ShadePixel
{
    uint id; // mesh + primitive IDs
    uint sampleMask;
};

//#define SORT_SIZE AA_SAMPLES
//#define SORT_T ShadePixel
//#define SORT_CMP_LESS(a, b) (a.id < b.id)
#define SORT_SIZE AA_SAMPLES
#define SORT_T uint
#define SORT_CMP_LESS(a, b) (a < b)
#include "Sort.h"

groupshared float3 tileFramebuffer[TILE_SIZE];

groupshared uint tileQuadCount;
groupshared uint tileDoneSlot;

#if QUAD_READ_GROUPSHARED_FALLBACK
groupshared float2 qr_float2[TILE_SIZE];
#endif

float3 ShadeQuadThread(
    uint threadID, // for groupshared fallback
    float3 rayDir, uint meshID, uint primID, float3 uvw
)
{
    uint materialID = g_meshInfo[meshID].materialID;
    TriInterpolated tri = triFetchAndInterpolate(meshID, primID, uvw);

#if QUAD_READ_GROUPSHARED_FALLBACK
    qr_float2[threadID] = tri.uv;
    GroupMemoryBarrierWithGroupSync();
    uint threadID00 = threadID & ~(QUAD_SIZE - 1);
    float2 uv00 = qr_float2[threadID00 + 0];
    float2 uv10 = qr_float2[threadID00 + 1];
    float2 uv01 = qr_float2[threadID00 + 2];
    GroupMemoryBarrierWithGroupSync();
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

[numthreads(TILE_SIZE, 1, 1)]
[RootSignature(
    "CBV(b0),"
    "CBV(b1),"
    "DescriptorTable(UAV(u2, numDescriptors = 4)),"
    "DescriptorTable(SRV(t1, numDescriptors = 3)),"
    "DescriptorTable(SRV(t100, numDescriptors = unbounded)),"
    "StaticSampler(s0, maxAnisotropy = 8),"
)]
void BeamsQuadShade(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileX = groupID.x;
    uint tileY = groupID.y;
    uint tileIndex = tileY * dynamicConstants.tilesX + tileX;
    uint threadID = groupThreadID.x;
    uint quadLocalIndex = threadID & (QUAD_SIZE - 1);
    //uint quadIndex = threadID / QUAD_SIZE;
    //uint quadLaneMask = (QUAD_SIZE - 1) << (quadIndex * QUAD_SIZE_LOG2);
    //uint localX;
    //uint localY;
    //threadIndexToQuadSwizzle(threadID, localX, localY);
    //uint pixelX = tileX * TILE_DIM_X + localX;
    //uint pixelY = tileY * TILE_DIM_Y + localY;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM_X;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM_Y;

    float3 tileFill = float3(0, 0, 0);
    bool tileOK = true;
    uint tileTriCount = g_tileTriCounts[tileIndex];
    if (tileTriCount <= 0)
    {
        // no triangles overlap this tile
        tileFill = float3(0, 0, 1);
        tileOK = false;
    }
    else if (tileTriCount > TILE_MAX_TRIS)
    {
        // tile tri list overflowed
        tileFill = float3(1, 0, 0);
        tileOK = false;
    }

    tileFramebuffer[threadID] = tileFill;
    if (threadID == 0)
    {
        tileQuadCount = 0;
        tileDoneSlot = ~uint(0);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tileOK)
    {
        while (true)
        {
            uint inputSlot;
            InterlockedAdd(tileQuadCount, 1, inputSlot);
            if (inputSlot >= MAX_SHADE_QUADS_PER_TILE)
                break;
            ShadeQuad shadeQuad = g_tileShadeQuads[tileIndex].quads[inputSlot];

            bool tileDone = (shadeQuad.bits & (1 << (QUADS_PER_TILE_LOG2 + 1))) != 0;
            if (tileDone)
                tileDoneSlot = inputSlot;
            GroupMemoryBarrierWithGroupSync();
            if (tileDoneSlot < inputSlot)
                break;

            uint quadIndex = shadeQuad.bits & (QUADS_PER_TILE - 1);
            bool quadDone = (shadeQuad.bits & (1 << (QUADS_PER_TILE_LOG2 + 0))) != 0;
            uint sampleCount = (shadeQuad.bits >> (QUADS_PER_TILE_LOG2 + 2 + AA_SAMPLES_LOG2 * quadLocalIndex)) & (AA_SAMPLES - 1);

            uint tileLocalIndex = quadIndex * QUAD_SIZE + quadLocalIndex;

            uint quadX = quadIndex % QUADS_PER_TILE_X;
            uint quadY = quadIndex / QUADS_PER_TILE_X;
            uint quadLocalX = quadLocalIndex % QUAD_DIM_X;
            uint quadLocalY = quadLocalIndex / QUAD_DIM_X;
            uint pixelX = tileX * TILE_DIM_X + quadX * QUAD_DIM_X + quadLocalX;
            uint pixelY = tileY * TILE_DIM_Y + quadY * QUAD_DIM_Y + quadLocalY;

            // TODO: centroid
            float3 rayOriginShade;
            float3 rayDirShade;
            GenerateCameraRay(
                uint2(pixelDimX, pixelDimY),
                float2(pixelX, pixelY),
                rayOriginShade, rayDirShade);

            uint id = shadeQuad.id;
            if (id == BAD_TRI_ID)
            {
                // beam traversal found potential triangles for this tile, but none survived past quad visibility test
                continue;
            }

            uint meshID = id >> 16;
            uint primID = id & 0xffff;
            Triangle tri = triFetch(meshID, primID);

            float3 uvw = triIntersectNoFail(rayOriginShade, rayDirShade, tri).xyz;

            tileFramebuffer[tileLocalIndex] += sampleCount * ShadeQuadThread(
                threadID,
                rayDirShade, meshID, primID, uvw);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint2 outputPos = uint2(
        tileX * TILE_DIM_X + threadID % TILE_DIM_X,
        tileY * TILE_DIM_Y + threadID / TILE_DIM_X);
    g_screenOutput[outputPos] = float4(tileFramebuffer[threadID] / AA_SAMPLES, 1);
}

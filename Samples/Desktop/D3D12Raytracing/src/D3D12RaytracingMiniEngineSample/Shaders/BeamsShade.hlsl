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

// Enable this to pack colors down into a uint, makes it easier to do atomic framebuffer updates
// when multiple threads could be shading quads that target the same pixel location.
// (Ideally, we'd have access to float atomics in HLSL and not need this)
#define PACKED_TILE_FB 1

#if PACKED_TILE_FB
# define TILE_FRAMEBUFFER_BITS 24
# define TILE_FRAMEBUFFER_MASK ((1 << TILE_FRAMEBUFFER_BITS) - 1)
# define TILE_FRAMEBUFFER_SCALE ((1 << (TILE_FRAMEBUFFER_BITS - AA_SAMPLES_LOG2)) - 1)
groupshared uint3 tileFramebufferPacked[TILE_SIZE];

uint3 tileFbPack(float3 c)
{
    return uint3(
        uint(c.x * TILE_FRAMEBUFFER_SCALE + .5f),
        uint(c.y * TILE_FRAMEBUFFER_SCALE + .5f),
        uint(c.z * TILE_FRAMEBUFFER_SCALE + .5f));
}

float3 tileFbUnpack(uint3 c)
{
    return float3(
        float(c.x & TILE_FRAMEBUFFER_MASK),
        float(c.y & TILE_FRAMEBUFFER_MASK),
        float(c.z & TILE_FRAMEBUFFER_MASK)) / (TILE_FRAMEBUFFER_SCALE * AA_SAMPLES);
}
#else
groupshared float3 tileFramebuffer[TILE_SIZE];
#endif

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

    float shadow = 1.0f;

    float3 outputColor = Shade(
        diffuseColor,
        shadeConstants.ambientColor,
        float3(.56f, .56f, .56f),
        specularMask,
        gloss,
        normal,
        viewDir,
        shadeConstants.sunDirection,
        shadeConstants.sunColor,
        shadow);

    return outputColor;
}

[numthreads(TILE_SIZE, 1, 1)]
[RootSignature(
    "CBV(b0),"
    "CBV(b1),"
    "DescriptorTable(UAV(u2, numDescriptors = 6)),"
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
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM_X;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM_Y;

    if (threadID == 0) PERF_COUNTER(shadeTiles, 1);

    uint2 outputPos = uint2(
        tileX * TILE_DIM_X + threadID % TILE_DIM_X,
        tileY * TILE_DIM_Y + threadID / TILE_DIM_X);

    uint quadCount = g_tileShadeQuadsCount[tileIndex];
    if (quadCount == 0)
    {
        // no shade quads generated for this tile
        if (threadID == 0) PERF_COUNTER(shadeNoQuads, 1);
        g_screenOutput[outputPos] = float4(0, 0, 1, 1);
        return;
    }
    else if (quadCount > MAX_SHADE_QUADS_PER_TILE)
    {
        // shade quad list overflowed
        if (threadID == 0) PERF_COUNTER(shadeOverflow, 1);
        g_screenOutput[outputPos] = float4(1, 0, 0, 1);
        return;
    }

#if PACKED_TILE_FB
    tileFramebufferPacked[threadID] = tileFbPack(float3(0, 0, 0));
#else
    tileFramebuffer[threadID] = tileFill;
#endif
    GroupMemoryBarrierWithGroupSync();

    uint inputSlot = threadID / QUAD_SIZE;
    while (inputSlot < quadCount)
    {
        if (quadLocalIndex == 0) PERF_COUNTER(shadeQuads, 1);

        ShadeQuad shadeQuad = g_tileShadeQuads[tileIndex].quads[inputSlot];
        inputSlot += QUADS_PER_TILE;

        uint quadIndex = shadeQuad.bits & (QUADS_PER_TILE - 1);
        uint sampleCount = shadeQuad.bits >> (QUADS_PER_TILE_LOG2 + (AA_SAMPLES_LOG2 + 1) * quadLocalIndex);
        sampleCount &= (1 << (AA_SAMPLES_LOG2 + 1)) - 1;

        uint fauxThreadID = quadIndex * QUAD_SIZE + quadLocalIndex;

        uint localX;
        uint localY;
        threadIndexToQuadSwizzle(fauxThreadID, localX, localY);
        uint tileFbIndex = localY * TILE_DIM_X + localX;
        uint pixelX = tileX * TILE_DIM_X + localX;
        uint pixelY = tileY * TILE_DIM_Y + localY;

        // TODO: centroid
        float3 rayOriginShade;
        float3 rayDirShade;
        GenerateCameraRay(
            uint2(pixelDimX, pixelDimY),
            float2(pixelX, pixelY),
            rayOriginShade, rayDirShade);

        uint id = shadeQuad.id;
        uint meshID = id >> PRIM_ID_BITS;
        uint primID = id & PRIM_ID_MASK;
        Triangle tri = triFetch(meshID, primID);

        float3 uvw = triIntersectNoFail(rayOriginShade, rayDirShade, tri).xyz;

#if PACKED_TILE_FB
        float3 shadeColor = ShadeQuadThread(
            threadID,
            rayDirShade, meshID, primID, uvw);

        shadeColor = clamp(shadeColor, 0.0f, 1.0f);

        uint3 packedColor = tileFbPack(shadeColor);
        packedColor *= sampleCount;

        InterlockedAdd(tileFramebufferPacked[tileFbIndex].x, packedColor.x);
        InterlockedAdd(tileFramebufferPacked[tileFbIndex].y, packedColor.y);
        InterlockedAdd(tileFramebufferPacked[tileFbIndex].z, packedColor.z);
#else
        tileFramebuffer[tileFbIndex] += sampleCount * ShadeQuadThread(
            threadID,
            rayDirShade, meshID, primID, uvw);
#endif
    }
    GroupMemoryBarrierWithGroupSync();

#if PACKED_TILE_FB
    float3 unpackedColor = tileFbUnpack(tileFramebufferPacked[threadID]);
    g_screenOutput[outputPos] = float4(unpackedColor, 1);
#else
    g_screenOutput[outputPos] = float4(tileFramebuffer[threadID] / AA_SAMPLES, 1);
#endif
}

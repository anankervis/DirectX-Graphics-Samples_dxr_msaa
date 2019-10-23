#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "Shading.h"
#include "TriFetch.h"

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

Texture2D<float4> g_materialTextures[] : register(t100);

groupshared float2 tileUVs[TILE_SIZE];

float4 Shade(
    uint threadID, // for groupshared fallback
    uint pixelDimX, uint pixelDimY, uint pixelX, uint pixelY,
    float3 rayOrigin, float3 rayDir, uint meshID, uint primID, float4 uvwt
)
{
    uint materialID = g_meshInfo[meshID].materialID;
    TriInterpolated tri = triFetchAndInterpolate(meshID, primID, uvwt.xyz);

    // Bug: well, no surprise, these don't work in compute shaders, even though they were spec'd to.
    // Response from Tex at MSFT: Known issue that will be fixed in Vibranium (Spring 2020).
    // Current GitHub master build has this fixed, but you still need a new DXIL.dll for signed shaders
    // for the whole thing to work...
    //float2 uv00 = QuadReadLaneAt(tri.uv, 0);
    //float2 uv10 = QuadReadLaneAt(tri.uv, 1);
    //float2 uv01 = QuadReadLaneAt(tri.uv, 2);

    // groupshared fallback
    tileUVs[threadID] = tri.uv;
    GroupMemoryBarrierWithGroupSync();
    uint threadID00 = threadID & ~3;
    float2 uv00 = tileUVs[threadID00 + 0];
    float2 uv10 = tileUVs[threadID00 + 1];
    float2 uv01 = tileUVs[threadID00 + 2];

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

    float3 viewDir = normalize(-rayDir);
    float specularMask = 0; // TODO: read the texture

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

    return float4(outputColor, 1);
}

// this swizzling enables the use of QuadRead* lane sharing intrinsics in a compute shader...
// ...if the intrinsics weren't currently disallowed (known issue, see above)
void threadIndexToQuadSwizzle(uint threadID, out uint localX, out uint localY)
{
    // address bit layout (high to low) for an 8x8 tile: yyxxyx
    uint tileDimMask = (1 << TILE_DIM_LOG2) - 1;
    localX = (((threadID >> 2                  ) << 1) | ( threadID       & 1)) & tileDimMask;
    localY = (((threadID >> (1 + TILE_DIM_LOG2)) << 1) | ((threadID >> 1) & 1)) & tileDimMask;
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
    uint pixelX = tileX * TILE_DIM + localX;
    uint pixelY = tileY * TILE_DIM + localY;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM;

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

    float3 rayOrigin;
    float3 rayDir;
    GenerateCameraRay(uint2(pixelDimX, pixelDimY), uint2(pixelX, pixelY), rayOrigin, rayDir);

    float4 nearestUVWT = FLT_MAX;
    uint nearestID = ~uint(0);
    for (uint tileTriIndex = 0; tileTriIndex < tileTriCount; tileTriIndex++)
    {
        uint id = g_tileTris[tileIndex].id[tileTriIndex];
        uint meshID = id >> 16;
        uint primID = id & 0xffff;

        Triangle tri = triFetch(meshID, primID);

        float4 uvwt = triIntersect(rayOrigin, rayDir, tri);

        if (uvwt.x >= 0 && uvwt.y >= 0 && uvwt.z >= 0 &&
            uvwt.w < nearestUVWT.w)
        {
            nearestUVWT = uvwt;
            nearestID = id;
        }
    }

    if (nearestID == ~uint(0))
    {
        // no hit
        g_screenOutput[uint2(pixelX, pixelY)] = float4(1, 0, 1, 1);
        return;
    }

    uint meshID = nearestID >> 16;
    uint primID = nearestID & 0xffff;
    float4 outColor = Shade(
        threadID,
        pixelDimX, pixelDimY, pixelX, pixelY,
        rayOrigin, rayDir, meshID, primID, nearestUVWT);

    g_screenOutput[uint2(pixelX, pixelY)] = outColor;
}

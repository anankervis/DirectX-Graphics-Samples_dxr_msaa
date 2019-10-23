#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "Shading.h"

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

Texture2D<float4> g_materialTextures[] : register(t100);

float4 Shade(
    uint pixelDimX, uint pixelDimY, uint pixelX, uint pixelY,
    float3 rayOrigin, float3 rayDir, uint meshID, uint primID, float4 uvwt
)
{
    RayTraceMeshInfo mesh = g_meshInfo[meshID];
    uint materialID = mesh.materialID;

    const uint3 ii = Load3x16BitIndices(mesh.indexOffset + primID * 3 * 2);
    const float2 uv0 = GetUVAttribute(mesh.attrOffsetTexcoord0 + ii.x * mesh.attrStride);
    const float2 uv1 = GetUVAttribute(mesh.attrOffsetTexcoord0 + ii.y * mesh.attrStride);
    const float2 uv2 = GetUVAttribute(mesh.attrOffsetTexcoord0 + ii.z * mesh.attrStride);

    float3 bary = uvwt.xyz;
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    const float3 normal0 = asfloat(g_attributes.Load3(mesh.attrOffsetNormal + ii.x * mesh.attrStride));
    const float3 normal1 = asfloat(g_attributes.Load3(mesh.attrOffsetNormal + ii.y * mesh.attrStride));
    const float3 normal2 = asfloat(g_attributes.Load3(mesh.attrOffsetNormal + ii.z * mesh.attrStride));
    float3 vsNormal = normalize(normal0 * bary.x + normal1 * bary.y + normal2 * bary.z);

    const float3 tangent0 = asfloat(g_attributes.Load3(mesh.attrOffsetTangent + ii.x * mesh.attrStride));
    const float3 tangent1 = asfloat(g_attributes.Load3(mesh.attrOffsetTangent + ii.y * mesh.attrStride));
    const float3 tangent2 = asfloat(g_attributes.Load3(mesh.attrOffsetTangent + ii.z * mesh.attrStride));
    float3 vsTangent = normalize(tangent0 * bary.x + tangent1 * bary.y + tangent2 * bary.z);

    // Reintroduced the bitangent because we aren't storing the handedness of the tangent frame anywhere.  Assuming the space
    // is right-handed causes normal maps to invert for some surfaces.  The Sponza mesh has all three axes of the tangent frame.
    //float3 vsBitangent = normalize(cross(vsNormal, vsTangent)) * (isRightHanded ? 1.0 : -1.0);
    const float3 bitangent0 = asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + ii.x * mesh.attrStride));
    const float3 bitangent1 = asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + ii.y * mesh.attrStride));
    const float3 bitangent2 = asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + ii.z * mesh.attrStride));
    float3 vsBitangent = normalize(bitangent0 * bary.x + bitangent1 * bary.y + bitangent2 * bary.z);

    // TODO: Should just store uv partial derivatives in here rather than loading position and caculating it per pixel
    const float3 p0 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + ii.x * mesh.attrStride));
    const float3 p1 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + ii.y * mesh.attrStride));
    const float3 p2 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + ii.z * mesh.attrStride));

    float3 worldPosition = rayOrigin + rayDir * uvwt.w;

    float3 ddxOrigin, ddxDir, ddyOrigin, ddyDir;
    GenerateCameraRay(uint2(pixelDimX, pixelDimY), uint2(pixelX + 1, pixelY), ddxOrigin, ddxDir);
    GenerateCameraRay(uint2(pixelDimX, pixelDimY), uint2(pixelX, pixelY + 1), ddyOrigin, ddyDir);

    float3 triangleNormal = normalize(cross(p2 - p0, p1 - p0));
    float3 xOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddxOrigin, ddxDir);
    float3 yOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddyOrigin, ddyDir);

    float3 dpdu, dpdv;
    CalculateTrianglePartialDerivatives(uv0, uv1, uv2, p0, p1, p2, dpdu, dpdv);
    float2 ddx, ddy;
    CalculateUVDerivatives(triangleNormal, dpdu, dpdv, worldPosition, xOffsetPoint, yOffsetPoint, ddx, ddy);

    const float3 viewDir = normalize(-rayDir);

    const float3 diffuseColor = g_materialTextures[materialID * 2 + 0].SampleGrad(g_s0, uv, ddx, ddy).rgb;
    float specularMask = 0;     // TODO: read the texture

    float gloss = 128.0;
    float3 normal = g_materialTextures[materialID * 2 + 1].SampleGrad(g_s0, uv, ddx, ddy).rgb * 2.0 - 1.0;
    AntiAliasSpecular(normal, gloss);
    float3x3 tbn = float3x3(vsTangent, vsBitangent, vsNormal);
    normal = normalize(mul(normal, tbn));

    float3 outputColor = Shade(
        diffuseColor,
        shadeConstants.ambientColor,
        float3(0.56, 0.56, 0.56),
        specularMask,
        gloss,
        normal,
        viewDir,
        shadeConstants.sunDirection,
        shadeConstants.sunColor);

    return float4(outputColor, 1);
}

[numthreads(TILE_DIM, TILE_DIM, 1)]
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
    uint localX = groupThreadID.x;
    uint localY = groupThreadID.y;
    uint pixelX = dispatchThreadID.x;
    uint pixelY = dispatchThreadID.y;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM;

    uint tileTriCount = g_tileTriCounts[tileIndex];
    if (tileTriCount <= 0)
    {
        // no triangles overlap this tile
        g_screenOutput[dispatchThreadID.xy] = float4(0, 0, 1, 1);
        return;
    }
    else if (tileTriCount > TILE_MAX_TRIS)
    {
        // tile tri list overflowed
        g_screenOutput[dispatchThreadID.xy] = float4(1, 0, 0, 1);
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

        RayTraceMeshInfo mesh = g_meshInfo[meshID];

        uint3 indices = Load3x16BitIndices(mesh.indexOffset + primID * 3 * 2);

        Triangle tri;
        tri.v0 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.x * mesh.attrStride));
        tri.v1 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.y * mesh.attrStride));
        tri.v2 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.z * mesh.attrStride));

        float4 uvwt = TestRayTriangle(rayOrigin, rayDir, tri);

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
        g_screenOutput[dispatchThreadID.xy] = float4(1, 0, 1, 1);
        return;
    }

    uint meshID = nearestID >> 16;
    uint primID = nearestID & 0xffff;
    float4 outColor = Shade(pixelDimX, pixelDimY, pixelX, pixelY, rayOrigin, rayDir, meshID, primID, nearestUVWT);

    g_screenOutput[dispatchThreadID.xy] = outColor;
}

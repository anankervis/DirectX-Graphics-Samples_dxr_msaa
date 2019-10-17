//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):    James Stanard, Christopher Wallis
//

#define HLSL

#include "RayCommon.h"
#include "Shading.h"

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

[shader("closesthit")]
void Hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint meshID = rootConstants.meshID;

    RayTraceMeshInfo info = g_meshInfo[meshID];

    const uint3 ii = Load3x16BitIndices(info.indexOffset + PrimitiveIndex() * 3 * 2);
    const float2 uv0 = GetUVAttribute(info.attrOffsetTexcoord0 + ii.x * info.attrStride);
    const float2 uv1 = GetUVAttribute(info.attrOffsetTexcoord0 + ii.y * info.attrStride);
    const float2 uv2 = GetUVAttribute(info.attrOffsetTexcoord0 + ii.z * info.attrStride);

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    const float3 normal0 = asfloat(g_attributes.Load3(info.attrOffsetNormal + ii.x * info.attrStride));
    const float3 normal1 = asfloat(g_attributes.Load3(info.attrOffsetNormal + ii.y * info.attrStride));
    const float3 normal2 = asfloat(g_attributes.Load3(info.attrOffsetNormal + ii.z * info.attrStride));
    float3 vsNormal = normalize(normal0 * bary.x + normal1 * bary.y + normal2 * bary.z);

    const float3 tangent0 = asfloat(g_attributes.Load3(info.attrOffsetTangent + ii.x * info.attrStride));
    const float3 tangent1 = asfloat(g_attributes.Load3(info.attrOffsetTangent + ii.y * info.attrStride));
    const float3 tangent2 = asfloat(g_attributes.Load3(info.attrOffsetTangent + ii.z * info.attrStride));
    float3 vsTangent = normalize(tangent0 * bary.x + tangent1 * bary.y + tangent2 * bary.z);

    // Reintroduced the bitangent because we aren't storing the handedness of the tangent frame anywhere.  Assuming the space
    // is right-handed causes normal maps to invert for some surfaces.  The Sponza mesh has all three axes of the tangent frame.
    //float3 vsBitangent = normalize(cross(vsNormal, vsTangent)) * (isRightHanded ? 1.0 : -1.0);
    const float3 bitangent0 = asfloat(g_attributes.Load3(info.attrOffsetBitangent + ii.x * info.attrStride));
    const float3 bitangent1 = asfloat(g_attributes.Load3(info.attrOffsetBitangent + ii.y * info.attrStride));
    const float3 bitangent2 = asfloat(g_attributes.Load3(info.attrOffsetBitangent + ii.z * info.attrStride));
    float3 vsBitangent = normalize(bitangent0 * bary.x + bitangent1 * bary.y + bitangent2 * bary.z);

    // TODO: Should just store uv partial derivatives in here rather than loading position and caculating it per pixel
    const float3 p0 = asfloat(g_attributes.Load3(info.attrOffsetPos + ii.x * info.attrStride));
    const float3 p1 = asfloat(g_attributes.Load3(info.attrOffsetPos + ii.y * info.attrStride));
    const float3 p2 = asfloat(g_attributes.Load3(info.attrOffsetPos + ii.z * info.attrStride));

    float3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    uint2 threadID = DispatchRaysIndex().xy;
    float3 ddxOrigin, ddxDir, ddyOrigin, ddyDir;
    GenerateCameraRay(DispatchRaysDimensions().xy, uint2(threadID.x + 1, threadID.y), ddxOrigin, ddxDir);
    GenerateCameraRay(DispatchRaysDimensions().xy, uint2(threadID.x, threadID.y + 1), ddyOrigin, ddyDir);

    float3 triangleNormal = normalize(cross(p2 - p0, p1 - p0));
    float3 xOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddxOrigin, ddxDir);
    float3 yOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddyOrigin, ddyDir);

    float3 dpdu, dpdv;
    CalculateTrianglePartialDerivatives(uv0, uv1, uv2, p0, p1, p2, dpdu, dpdv);
    float2 ddx, ddy;
    CalculateUVDerivatives(triangleNormal, dpdu, dpdv, worldPosition, xOffsetPoint, yOffsetPoint, ddx, ddy);

    const float3 viewDir = normalize(-WorldRayDirection());

    const float3 diffuseColor = g_localTexture.SampleGrad(g_s0, uv, ddx, ddy).rgb;
    float specularMask = 0;     // TODO: read the texture

    float gloss = 128.0;
    float3 normal = g_localNormal.SampleGrad(g_s0, uv, ddx, ddy).rgb * 2.0 - 1.0;
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

    g_screenOutput[DispatchRaysIndex().xy] = float4(outputColor, 1);
}

[shader("miss")]
void Miss(inout RayPayload payload)
{
    g_screenOutput[DispatchRaysIndex().xy] = float4(0, 0, 0, 1);
}

[shader("raygeneration")]
void RayGen()
{
    float3 origin, direction;
    GenerateCameraRay(DispatchRaysDimensions().xy, DispatchRaysIndex().xy, origin, direction);

    RayDesc rayDesc =
    {
        origin,
        0.0f,
        direction,
        FLT_MAX
    };

    RayPayload payload;

    TraceRay(g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
}

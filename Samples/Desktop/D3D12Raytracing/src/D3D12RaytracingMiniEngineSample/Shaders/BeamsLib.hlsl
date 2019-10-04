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
void Hit(inout BeamPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint materialID = rootConstants.materialID;
    uint triangleID = PrimitiveIndex();

    RayTraceMeshInfo info = g_meshInfo[materialID];

    const uint3 ii = Load3x16BitIndices(info.m_indexOffsetBytes + PrimitiveIndex() * 3 * 2);
    const float2 uv0 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes);
    const float2 uv1 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes);
    const float2 uv2 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes);

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    const float3 normal0 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 normal1 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 normal2 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    float3 vsNormal = normalize(normal0 * bary.x + normal1 * bary.y + normal2 * bary.z);

    const float3 tangent0 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 tangent1 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 tangent2 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    float3 vsTangent = normalize(tangent0 * bary.x + tangent1 * bary.y + tangent2 * bary.z);

    // Reintroduced the bitangent because we aren't storing the handedness of the tangent frame anywhere.  Assuming the space
    // is right-handed causes normal maps to invert for some surfaces.  The Sponza mesh has all three axes of the tangent frame.
    //float3 vsBitangent = normalize(cross(vsNormal, vsTangent)) * (isRightHanded ? 1.0 : -1.0);
    const float3 bitangent0 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 bitangent1 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 bitangent2 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    float3 vsBitangent = normalize(bitangent0 * bary.x + bitangent1 * bary.y + bitangent2 * bary.z);

    // TODO: Should just store uv partial derivatives in here rather than loading position and caculating it per pixel
    const float3 p0 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 p1 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 p2 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));

    float3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    uint2 threadID = DispatchRaysIndex().xy;
    float3 ddxOrigin, ddxDir, ddyOrigin, ddyDir;
    GenerateCameraRay(uint2(threadID.x + 1, threadID.y), ddxOrigin, ddxDir);
    GenerateCameraRay(uint2(threadID.x, threadID.y + 1), ddyOrigin, ddyDir);

    float3 triangleNormal = normalize(cross(p2 - p0, p1 - p0));
    float3 xOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddxOrigin, ddxDir);
    float3 yOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddyOrigin, ddyDir);

    float3 dpdu, dpdv;
    CalculateTrianglePartialDerivatives(uv0, uv1, uv2, p0, p1, p2, dpdu, dpdv);
    float2 ddx, ddy;
    CalculateUVDerivatives(triangleNormal, dpdu, dpdv, worldPosition, xOffsetPoint, yOffsetPoint, ddx, ddy);

    const float3 viewDir = normalize(-WorldRayDirection());
    uint materialInstanceId = info.m_materialInstanceId;

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

/*
Record the triangle and move on. We'll compute coverage later.

Yes, it would be nice to have per-sample coverage auto-generated and fed into this function...
...but how would that work with beams?

Single-pixel rays w/ MSAA can be treated as a collection of 2, 4, 8, 16 subrays w/ coverage mask.
Beams represent the conservative volume, which may be a collection of pixels or a spatial query.
*/
[shader("anyhit")]
void AnyHit(inout BeamPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
}

[shader("miss")]
void Miss(inout BeamPayload payload)
{
    g_screenOutput[DispatchRaysIndex().xy] = float4(0, 0, 0, 1);
}

[shader("raygeneration")]
void RayGen()
{
    float3 origin, direction;
    GenerateCameraRay(DispatchRaysIndex().xy, origin, direction);

    RayDesc rayDesc =
    {
        origin,
        0.0f,
        direction,
        FLT_MAX
    };

    BeamPayload payload;

    TraceRay(g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
}

/*
Convert beam triangle list to quads of triangleID + coverage mask
*/
[numthreads(BEAM_SIZE, BEAM_SIZE, 1)]
[RootSignature(
    "UAV(u2)"
)]
void BeamTrisToQuads(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    g_screenOutput[dispatchThreadID.xy] = float4(0, 1, 0, 1);
}

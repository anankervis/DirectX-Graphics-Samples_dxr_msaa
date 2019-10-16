#pragma once

#ifndef HLSL
# include "HlslCompat.h"
#endif

#define TILE_SIZE 2
#define TILE_MAX_TRIS 16

struct TileTri
{
    uint id[TILE_MAX_TRIS]; // material + primitive IDs
};

struct RayTraceMeshInfo
{
    uint  m_indexOffsetBytes;
    uint  m_uvAttributeOffsetBytes;
    uint  m_normalAttributeOffsetBytes;
    uint  m_tangentAttributeOffsetBytes;
    uint  m_bitangentAttributeOffsetBytes;
    uint  m_positionAttributeOffsetBytes;
    uint  m_attributeStrideBytes;
    uint  m_materialInstanceId;
};

// Volatile part (can be split into its own CBV). 
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3   worldCameraPosition;
    uint     padding;

    uint tilesX;
    uint tilesY;
};

struct RootConstants
{
    uint materialID;
};

struct RayPayload
{
    uint pad;
};

struct BeamPayload
{
    uint pad;
};

#ifdef HLSL

# ifndef SINGLE
static const float FLT_MAX = asfloat(0x7F7FFFFF);
# endif

RaytracingAccelerationStructure g_accel : register(t0);

StructuredBuffer<RayTraceMeshInfo> g_meshInfo : register(t1);
ByteAddressBuffer g_indices : register(t2);
ByteAddressBuffer g_attributes : register(t3);

Texture2D<float4> g_localTexture : register(t10);
Texture2D<float4> g_localNormal : register(t11);

SamplerState      g_s0 : register(s0);

RWTexture2D<float4> g_screenOutput : register(u2);
RWStructuredBuffer<uint> g_tileTriCounts : register(u3);
RWStructuredBuffer<TileTri> g_tileTris : register(u4);

cbuffer b1 : register(b1)
{
    DynamicCB dynamicConstants;
};

cbuffer b3 : register(b3)
{
    RootConstants rootConstants;
};

uint3 Load3x16BitIndices(
    uint offsetBytes)
{
    const uint dwordAlignedOffset = offsetBytes & ~3;

    const uint2 four16BitIndices = g_indices.Load2(dwordAlignedOffset);

    uint3 indices;

    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}

float2 GetUVAttribute(uint byteOffset)
{
    return asfloat(g_attributes.Load2(byteOffset));
}

inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5; // center in the middle of the pixel
    float2 screenPos = xy / float2(DispatchRaysDimensions().xy) * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    // Unproject into a ray
    float4 unprojected = mul(dynamicConstants.cameraToWorld, float4(screenPos, 0, 1));
    float3 world = unprojected.xyz / unprojected.w;
    origin = dynamicConstants.worldCameraPosition;
    direction = normalize(world - origin);
}

float3 RayPlaneIntersection(float3 planeOrigin, float3 planeNormal, float3 rayOrigin, float3 rayDirection)
{
    float t = dot(-planeNormal, rayOrigin - planeOrigin) / dot(planeNormal, rayDirection);
    return rayOrigin + rayDirection * t;
}

bool Inverse2x2(float2x2 mat, out float2x2 inverse)
{
    float determinant = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    float rcpDeterminant = rcp(determinant);
    inverse[0][0] = mat[1][1];
    inverse[1][1] = mat[0][0];
    inverse[1][0] = -mat[0][1];
    inverse[0][1] = -mat[1][0];
    inverse = rcpDeterminant * inverse;

    return abs(determinant) > 0.00000001;
}

/* TODO: Could be precalculated per triangle
Using implementation described in PBRT, finding the partial derivative of the (change in position)/(change in UV coordinates)
a.k.a dp/du and dp/dv

Given the 3 UV and 3 triangle points, this can be represented as a linear equation:

(uv0.u - uv2.u, uv0.v - uv2.v)   (dp/du)   =     (p0 - p2)
(uv1.u - uv2.u, uv1.v - uv2.v)   (dp/dv)   =     (p1 - p2)

To solve for dp/du, we invert the 2x2 matrix on the left side to get

(dp/du)   = (uv0.u - uv2.u, uv0.v - uv2.v)^-1  (p0 - p2)
(dp/dv)   = (uv1.u - uv2.u, uv1.v - uv2.v)     (p1 - p2)
*/
void CalculateTrianglePartialDerivatives(float2 uv0, float2 uv1, float2 uv2, float3 p0, float3 p1, float3 p2, out float3 dpdu, out float3 dpdv)
{
    float2x2 linearEquation;
    linearEquation[0] = uv0 - uv2;
    linearEquation[1] = uv1 - uv2;

    float2x3 pointVector;
    pointVector[0] = p0 - p2;
    pointVector[1] = p1 - p2;
    float2x2 inverse;
    Inverse2x2(linearEquation, inverse);
    dpdu = pointVector[0] * inverse[0][0] + pointVector[1] * inverse[0][1];
    dpdv = pointVector[0] * inverse[1][0] + pointVector[1] * inverse[1][1];
}

/*
Using implementation described in PBRT, finding the derivative for the UVs (dU, dV)  in both the x and y directions

Given the original point and the offset points (pX and pY) + the partial derivatives, the linear equation can be formed:
Note described only with pX, but the same is also applied to pY

( dpdu.x, dpdv.x)          =   (pX.x - p.x)
( dpdu.y, dpdv.y)   (dU)   =   (pX.y - p.y)
( dpdu.z, dpdv.z)   (dV)   =   (pX.z - p.z)

Because the problem is over-constrained (3 equations and only 2 unknowns), we pick 2 channels, and solve for dU, dV by inverting the matrix

dU    =   ( dpdu.x, dpdv.x)^-1  (pX.x - p.x)
dV    =   ( dpdu.y, dpdv.y)     (pX.y - p.y)
*/

void CalculateUVDerivatives(float3 normal, float3 dpdu, float3 dpdv, float3 p, float3 pX, float3 pY, out float2 ddX, out float2 ddY)
{
    int2 indices;
    float3 absNormal = abs(normal);
    if (absNormal.x > absNormal.y && absNormal.x > absNormal.z)
    {
        indices = int2(1, 2);
    }
    else if (absNormal.y > absNormal.z)
    {
        indices = int2(0, 2);
    }
    else
    {
        indices = int2(0, 1);
    }

    float2x2 linearEquation;
    linearEquation[0] = float2(dpdu[indices.x], dpdv[indices.x]);
    linearEquation[1] = float2(dpdu[indices.y], dpdv[indices.y]);

    float2x2 inverse;
    Inverse2x2(linearEquation, inverse);
    float2 pointOffset = float2(pX[indices.x] - p[indices.x], pX[indices.y] - p[indices.y]);
    ddX = abs(mul(inverse, pointOffset));

    pointOffset = float2(pY[indices.x] - p[indices.x], pY[indices.y] - p[indices.y]);
    ddY = abs(mul(inverse, pointOffset));
}

#endif

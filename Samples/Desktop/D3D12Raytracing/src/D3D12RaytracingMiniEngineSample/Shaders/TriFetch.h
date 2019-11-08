#pragma once

#include "RayCommon.h"

struct Triangle
{
    float3 v0;
    float3 e0;
    float3 e1;
};

struct TriInterpolated
{
    float3 worldPos;
    float2 uv;
    float3 normal;
    float3 tangent;
    float3 bitangent;
};

uint3 triFetchIndices(uint offsetBytes)
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

Triangle triFetch(uint meshID, uint primID)
{
    RayTraceMeshInfo mesh = g_meshInfo[meshID];

    uint3 indices = triFetchIndices(mesh.indexOffset + primID * 3 * 2);

    Triangle tri;
    tri.v0 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.x * mesh.attrStride));
    float3 v1 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.y * mesh.attrStride));
    float3 v2 = asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.z * mesh.attrStride));

    tri.e0 = v1 - tri.v0;
    tri.e1 = v2 - tri.v0;

    return tri;
}

TriInterpolated triFetchAndInterpolate(uint meshID, uint primID, float3 uvw)
{
    RayTraceMeshInfo mesh = g_meshInfo[meshID];

    uint3 indices = triFetchIndices(mesh.indexOffset + primID * 3 * 2);

    TriInterpolated tri;
    tri.worldPos =
        uvw.x * asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.x * mesh.attrStride)) +
        uvw.y * asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.y * mesh.attrStride)) +
        uvw.z * asfloat(g_attributes.Load3(mesh.attrOffsetPos + indices.z * mesh.attrStride));

    tri.uv =
        uvw.x * asfloat(g_attributes.Load2(mesh.attrOffsetTexcoord0 + indices.x * mesh.attrStride)) +
        uvw.y * asfloat(g_attributes.Load2(mesh.attrOffsetTexcoord0 + indices.y * mesh.attrStride)) +
        uvw.z * asfloat(g_attributes.Load2(mesh.attrOffsetTexcoord0 + indices.z * mesh.attrStride));

    tri.normal =
        uvw.x * asfloat(g_attributes.Load3(mesh.attrOffsetNormal + indices.x * mesh.attrStride)) +
        uvw.y * asfloat(g_attributes.Load3(mesh.attrOffsetNormal + indices.y * mesh.attrStride)) +
        uvw.z * asfloat(g_attributes.Load3(mesh.attrOffsetNormal + indices.z * mesh.attrStride));

    tri.tangent =
        uvw.x * asfloat(g_attributes.Load3(mesh.attrOffsetTangent + indices.x * mesh.attrStride)) +
        uvw.y * asfloat(g_attributes.Load3(mesh.attrOffsetTangent + indices.y * mesh.attrStride)) +
        uvw.z * asfloat(g_attributes.Load3(mesh.attrOffsetTangent + indices.z * mesh.attrStride));

    tri.bitangent =
        uvw.x * asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + indices.x * mesh.attrStride)) +
        uvw.y * asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + indices.y * mesh.attrStride)) +
        uvw.z * asfloat(g_attributes.Load3(mesh.attrOffsetBitangent + indices.z * mesh.attrStride));

    return tri;
}

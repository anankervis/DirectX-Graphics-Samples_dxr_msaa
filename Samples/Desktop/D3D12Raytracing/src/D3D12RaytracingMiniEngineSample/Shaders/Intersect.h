#pragma once

#include "TriFetch.h"

// return value xyz are barycentrics, w is t
float4 triIntersect(float3 rayOrigin, float3 rayDir, Triangle tri)
{
    float3 v0 = tri.v0;
    float3 edge0 = tri.v1 - tri.v0;
    float3 edge1 = tri.v2 - tri.v0;

    float3 n = cross(edge0, edge1);

    float denom = dot(-rayDir, n);
    // ray is parallel to triangle, or fails backfacing test?
    if (denom <= 0.0f)
        return float4(-1.0f, -1.0f, -1.0f, -1.0f);
    float3 v0ToRayOrigin = rayOrigin - v0;
    float t = dot(v0ToRayOrigin, n);
    if (t < 0.0f) // intersection falls before ray origin?
        return float4(-1.0f, -1.0f, -1.0f, -1.0f);

    // compute barycentrics
    float3 e = cross(-rayDir, v0ToRayOrigin);
    float v = dot(edge1, e);
    float w = -dot(edge0, e);

    float ood = 1.0f / denom;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = 1.0f - v - w;

    return float4(u, v, w, t);
}

// We already know the ray will hit the triangle's plane, so don't check for failure cases.
// Used when calculating derivatives, or recomputing UVWT for shading the already-found nearest hit.
float4 triIntersectNoFail(float3 rayOrigin, float3 rayDir, Triangle tri)
{
    float3 v0 = tri.v0;
    float3 edge0 = tri.v1 - tri.v0;
    float3 edge1 = tri.v2 - tri.v0;

    float3 n = cross(edge0, edge1);

    float denom = dot(-rayDir, n);

    float3 v0ToRayOrigin = rayOrigin - v0;
    float t = dot(v0ToRayOrigin, n);

    // compute barycentrics
    float3 e = cross(-rayDir, v0ToRayOrigin);
    float v = dot(edge1, e);
    float w = -dot(edge0, e);

    float ood = 1.0f / denom;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = 1.0f - v - w;

    return float4(u, v, w, t);
}

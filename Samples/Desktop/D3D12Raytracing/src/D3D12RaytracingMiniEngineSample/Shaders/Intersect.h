#pragma once

#include "TriFetch.h"

// return value xyz are barycentrics, w is t
float4 triIntersect(float3 rayOrigin, float3 rayDir, Triangle tri)
{
    float3 n = cross(tri.e0, tri.e1);

    float denom = dot(-rayDir, n);
    // ray is parallel to triangle, or fails backfacing test?
    if (denom <= 0.0f)
        return float4(-1.0f, -1.0f, -1.0f, -1.0f);
    float3 v0ToRayOrigin = rayOrigin - tri.v0;
    float t = dot(v0ToRayOrigin, n);
    if (t < 0.0f) // intersection falls before ray origin?
        return float4(-1.0f, -1.0f, -1.0f, -1.0f);

    // compute barycentrics
    float3 e = cross(-rayDir, v0ToRayOrigin);
    float v = dot(tri.e1, e);
    float w = -dot(tri.e0, e);

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
    float3 n = cross(tri.e0, tri.e1);

    float denom = dot(-rayDir, n);

    float3 v0ToRayOrigin = rayOrigin - tri.v0;
    float t = dot(v0ToRayOrigin, n);

    // compute barycentrics
    float3 e = cross(-rayDir, v0ToRayOrigin);
    float v = dot(tri.e1, e);
    float w = -dot(tri.e0, e);

    float ood = 1.0f / denom;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = 1.0f - v - w;

    return float4(u, v, w, t);
}

struct Frustum
{
    enum { planeCount = 4 };

    // no near or far
    float4 p[planeCount];
};

Frustum FrustumCreate(float3 rayOrigin, float3 rayDirs[4])
{
    Frustum f;
    for (int n = 0; n < Frustum::planeCount; n++)
    {
        float3 dir = cross(
            rayDirs[(n + 0) % Frustum::planeCount],
            rayDirs[(n + 1) % Frustum::planeCount]);

        float dist = dot(rayOrigin, dir);

        f.p[n] = float4(dir, dist);
    }
    return f;
}

bool FrustumTest(Frustum f, Triangle tri)
{
    for (int n = 0; n < Frustum::planeCount; n++)
    {
        float3 dir = f.p[n].xyz;
        float dist = f.p[n].w;

        float d0 = dot(dir, tri.v0);
        float d1 = d0 + dot(dir, tri.e0);
        float d2 = d0 + dot(dir, tri.e1);

        if (d0 > dist && d1 > dist && d2 > dist)
        {
            return false;
        }
    }

    return true;
}

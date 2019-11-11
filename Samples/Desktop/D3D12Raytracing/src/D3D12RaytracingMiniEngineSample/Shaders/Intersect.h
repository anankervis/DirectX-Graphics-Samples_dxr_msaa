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

// return true if the triangle's UVW interval has partial or full overlap of the tile
// assumes tris where denom <= 0.0f and t < 0.0f have already been rejected
// TODO(anankervis): optimize this... if the whole tile shares a coord system, could compute
// min/max U, V, W directly from tile-uniform derivatives
bool FrustumTestUVW(float3 rayOrigin, float3 rayDirs[4], Triangle tri)
{
    float4 uvw[4];
    {for (int n = 0; n < 4; n++)
    {
        uvw[n] = triIntersectNoFail(rayOrigin, rayDirs[n], tri);
    }}

    float uMin = FLT_MAX;
    float uMax = -FLT_MAX;
    float vMin = FLT_MAX;
    float vMax = -FLT_MAX;
    float wMin = FLT_MAX;
    float wMax = -FLT_MAX;
    {for (int n = 0; n < 4; n++)
    {
        uMin = min(uMin, uvw[n].x);
        uMax = max(uMax, uvw[n].x);
        vMin = min(vMin, uvw[n].y);
        vMax = max(vMax, uvw[n].y);
        wMin = min(wMin, uvw[n].z);
        wMax = max(wMax, uvw[n].z);
    }}

    if (uMax < 0 || uMin > 1 || vMax < 0 || vMin > 1 || wMax < 0 || wMin > 1)
        return false; // all out
    //if (uMin < 0 || uMax > 1 || vMin < 0 || vMax > 1 || wMin < 0 || wMax > 1)
    //    return intersect_partial; // partial
    return true; // all in
}

struct TriTile
{
    float t;
    float3 e0;
    float3 e1;
    float3 v0ToRayOrigin;
};

bool TriTileSetup(Triangle tri, float3 rayOrigin, out TriTile triTile)
{
    triTile.e0 = tri.e0;
    triTile.e1 = tri.e1;

    float3 normal = cross(triTile.e0, triTile.e1);
    triTile.v0ToRayOrigin = rayOrigin - tri.v0;

    triTile.t = dot(triTile.v0ToRayOrigin, normal);

    // this test could be precomputed each frame for a pinhole camera, and failing triangles discarded earlier
    if (triTile.t < 0.0f)
        return false; // ray origin is behind the triangle's plane

    return true;
}

struct TriThread
{
    float denomCenter;
    float vCenter;
    float wCenter;
    float2 dDenomDAlpha;
    float2 dVdAlpha;
    float2 dWdAlpha;
};

void GetDifferentials(
    float3 edge0,
    float3 edge1,
    float3 v0ToRayOrigin,
    float3 majorDirDiff,
    float3 minorDirDiff,
    out float2 dDenomDAlpha,
    out float2 dVdAlpha,
    out float2 dWdAlpha)
{
    float3 normal = cross(edge0, edge1);
    dDenomDAlpha = float2(
        dot(-majorDirDiff, normal),
        dot(-minorDirDiff, normal));

    dVdAlpha =
        float2(
            dot(edge1, cross(-majorDirDiff, v0ToRayOrigin)),
            dot(edge1, cross(-minorDirDiff, v0ToRayOrigin)));

    dWdAlpha =
        float2(
            -dot(edge0, cross(-majorDirDiff, v0ToRayOrigin)),
            -dot(edge0, cross(-minorDirDiff, v0ToRayOrigin)));
}

TriThread TriThreadSetup(
    TriTile triTile,
    float3 rayDirCenter, float3 majorDirDiff, float3 minorDirDiff)
{
    TriThread triThread;

    float3 normal = cross(triTile.e0, triTile.e1);
    triThread.denomCenter = dot(-rayDirCenter, normal);

    // compute scaled barycentrics
    float3 eCenter = cross(-rayDirCenter, triTile.v0ToRayOrigin);
    triThread.vCenter = dot(triTile.e1, eCenter);
    triThread.wCenter = -dot(triTile.e0, eCenter);
    GetDifferentials(
        triTile.e0, triTile.e1, triTile.v0ToRayOrigin,
        majorDirDiff, minorDirDiff,
        triThread.dDenomDAlpha, triThread.dVdAlpha, triThread.dWdAlpha);

    return triThread;
}


// returns true if the ray intersects the triangle and the intersection distance is less than depth
// also updates the value of depth
bool TriThreadTest(TriTile triTile, TriThread triThread, float2 alpha, inout float depth)
{
    // it seems that the CUDA compiler is missing an opportunity to merge multiply + add across function calls into
    // FMA, so no call to dot product function here...
    // 2 FMA
    float denom = triThread.denomCenter + triThread.dDenomDAlpha.x * alpha.x + triThread.dDenomDAlpha.y * alpha.y;

    // t still needs to be divided by denom to get the correct distance
    // this is a combination of two tests:
    // 1) denom <= 0.0f // ray is parallel to triangle, or fails backfacing test
    // 2) tri.t >= depth * denom // failed depth test
    // tri.t is known to be >= 0.0f
    // depth is known to be >= 0.0f
    // we can safely test both conditions with only test #2
    // triTile.t >= depth * denom
    // 0.0f >= depth * denom - triTile.t
    // depth * denom - triTile.t < 0.0f
    // 1 FMA
    float depthDelta = depth * denom - triTile.t;

    // compute scaled barycentrics
    // 2 FMA
    float v = triThread.vCenter + triThread.dVdAlpha.x * alpha.x + triThread.dVdAlpha.y * alpha.y;
    // 2 FMA
    float w = triThread.wCenter + triThread.dWdAlpha.x * alpha.x + triThread.dWdAlpha.y * alpha.y;
    // 2 ADD
    float u = denom - v - w;

    // depth test from above, plus: u < 0.0f || v < 0.0f || w < 0.0f
    // 1 LOP3
    // 1 LOP
    //int test = __float_as_int(depthDelta) | __float_as_int(u) | __float_as_int(v) | __float_as_int(w);
    // 1 ISETP
    //if (test < 0)
    //    return false;
    if (depthDelta < 0.0f || u < 0.0f || v < 0.0f || w < 0.0f)
        return false;

    // 1 RCP
    // 1 MUL
    depth = triTile.t * (1.0f / denom);
    return true;
}

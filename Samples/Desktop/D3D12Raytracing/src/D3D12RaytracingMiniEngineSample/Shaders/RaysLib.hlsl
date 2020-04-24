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

#include "Intersect.h"
#include "RayCommon.h"
#include "RayGen.h"
#include "Shading.h"
#include "TriFetch.h"

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
};

struct ShadowPayload
{
    float opacity;
};

#if SHADOW_MODE == SHADOW_MODE_BEAM
struct ShadowHitAttribs
{
};

[shader("anyhit")]
void AnyHitShadow(inout ShadowPayload payload, in ShadowHitAttribs attr)
{
    PERF_COUNTER(shadowBeamAnyHitCount, 1);

    uint primID = PrimitiveIndex();

// TODO: choose major axis, integrate overlap of beam and aabb
    float opacity = g_aabbShadow_payload[primID].opacity.x;

    payload.opacity += opacity;
}

[shader("intersection")]
void IntersectionShadow()
{
    PERF_COUNTER(shadowBeamIntersectCount, 1);

    float tMax = RayTCurrent();

    ShadowHitAttribs hit;

    ReportHit(tMax, 0, hit);
}
#else
[shader("closesthit")]
void HitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PERF_COUNTER(shadowHitCount, 1);

    payload.opacity = 1.0f;
}
#endif

[shader("closesthit")]
void HitPrimary(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PERF_COUNTER(closestHitCount, 1);

    uint sampleIndex = payload.sampleIndex;

    float3 rayDir = WorldRayDirection();
    uint meshID = rootConstants.meshID;
    uint primID = PrimitiveIndex();

    float3 uvw = float3(
        1.0f - attr.barycentrics.x - attr.barycentrics.y,
        attr.barycentrics.x,
        attr.barycentrics.y);
    TriInterpolated tri = triFetchAndInterpolate(meshID, primID, uvw);

    // find the UVWs of +1 X and +1 Y pixels, then calculate texcoord derivatives with finite differencing
    float3 rayOriginDX, rayDirDX;
    float3 rayOriginDY, rayDirDY;
    GenerateCameraRay(
        DispatchRaysDimensions().xy,
        DispatchRaysIndex().xy + uint2(1, 0) + AA_SAMPLE_OFFSET_TABLE[sampleIndex] * float2(1, -1), // Y direction is flipped vs beam vis shader
        rayOriginDX,
        rayDirDX);
    GenerateCameraRay(
        DispatchRaysDimensions().xy,
        DispatchRaysIndex().xy + uint2(0, 1) + AA_SAMPLE_OFFSET_TABLE[sampleIndex] * float2(1, -1), // Y direction is flipped vs beam vis shader
        rayOriginDY,
        rayDirDY);

    Triangle triPos = triFetch(meshID, primID);
    float4 uvwtDX = triIntersectNoFail(rayOriginDX, rayDirDX, triPos);
    float4 uvwtDY = triIntersectNoFail(rayOriginDY, rayDirDY, triPos);

    // hopefully the optimizer doesn't completely give up on life at this point, this should just be some extra math
    // reusing the stuff we loaded in the first triFetchAndInterpolate
    TriInterpolated triDX = triFetchAndInterpolate(meshID, primID, uvwtDX.xyz);
    TriInterpolated triDY = triFetchAndInterpolate(meshID, primID, uvwtDY.xyz);

    float2 uvDx = triDX.uv - tri.uv;
    float2 uvDy = triDY.uv - tri.uv;
    // Our derivatives are calculated with a spacing of 1 pixel, adjust for effective resolution boost of SSAA.
    // This way we get the texture sharpness you'd expect from running at 4x, 8x, etc. resolution and filtering down.
    float ssaaMipScale = 1.0f / sqrt(AA_SAMPLES);
    uvDx *= ssaaMipScale;
    uvDy *= ssaaMipScale;

    float3 diffuseColor = g_localTexture.SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb;

    float3 normal = g_localNormal.SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb * 2 - 1;
    float gloss = 128;
    AntiAliasSpecular(normal, gloss);

    float3x3 tbn = float3x3(
        normalize(tri.tangent),
        normalize(tri.bitangent),
        normalize(tri.normal));
    normal = normalize(mul(normal, tbn));

    float3 viewDir = normalize(rayDir);
    float specularMask = .1; // TODO: read the texture

#if SHADOW_MODE != SHADOW_MODE_NONE
    float3 shadowTarget = AREA_LIGHT_CENTER;

    uint shadowSampleCount = SHADOW_SAMPLES;
/*# if SHADOW_MODE == SHADOW_MODE_SOFT
    // for soft shadows, scale the sample count by the area we're integrating over
    {
        float area = AREA_LIGHT_EXTENT.x * AREA_LIGHT_EXTENT.z;
        float distance = max(.01f, length(shadowTarget - tri.worldPos));

        float projectedArea = area / distance;

        float samples = projectedArea * 100.0f;

        shadowSampleCount = clamp(uint(samples + .5), 1, SHADOW_SAMPLES);
    }
# endif*/

    float opacity = 0.0f;
    for (uint shadowSampleIndex = 0; shadowSampleIndex < shadowSampleCount; shadowSampleIndex++)
    {
        ShadowPayload shadowPayload;
        shadowPayload.opacity = 0.0f;

#if SHADOW_MODE == SHADOW_MODE_SOFT
        // soft shadows
        // use the area light to constrain the sampling region
        // each shading sample will shoot a shadow ray distributed somewhere on the area light surface

        // seed = pixel coord, sample index, shadow sample index
        float2 st = shadowRandom(DispatchRaysIndex().xy, (sampleIndex << SHADOW_SAMPLES_LOG2) | shadowSampleIndex);

        float3 shadowSampleTarget = shadowTarget;
        shadowSampleTarget += float3(st.x, 0, 0) * (AREA_LIGHT_EXTENT).x;
        shadowSampleTarget += float3(0, 0, st.y) * (AREA_LIGHT_EXTENT).z;

        float3 shadowRayDir = normalize(shadowSampleTarget - tri.worldPos);
#elif SHADOW_MODE == SHADOW_MODE_BEAM
        // soft beam shadows
        // use the area light center
        float3 shadowRayDir = normalize(shadowTarget - tri.worldPos);
#else
        // hard shadows
        // use the directional light
        // each shading sample will shoot one shadow ray, so we'll get super-sampled anti-aliased hard shadows
        float3 shadowRayDir = shadeConstants.sunDirection;
#endif

        // ray would shoot away from the area light
        if (dot(shadowRayDir, normal) <= 0.0f)
            continue;

        float tMin = .01f;
        RayDesc shadowRayDesc =
        {
            tri.worldPos,
            tMin,
            shadowRayDir,
            FLT_MAX,
        };

        PERF_COUNTER(shadowLaunchCount, 1);
        TraceRay(
            g_accelShadow,
#if SHADOW_MODE == SHADOW_MODE_BEAM
            RAY_FLAG_NONE,
#else
            RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
#endif
            ~0,
            HIT_GROUP_SHADOW, HIT_GROUP_COUNT, HIT_GROUP_SHADOW,
            shadowRayDesc, shadowPayload);

        opacity += clamp(shadowPayload.opacity, 0.0f, 1.0f);
    }
    float shadow = 1.0f - opacity / shadowSampleCount;
#else
    float shadow = 1.0f;
#endif

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
//outputColor = float3(shadow, shadow, shadow);

    payload.color += outputColor;
}

[shader("miss")]
void MissPrimary(inout RayPayload payload)
{
    PERF_COUNTER(missCount, 1);

    g_screenOutput[DispatchRaysIndex().xy] = float4(0, 0, 0, 1);
}

[shader("raygeneration")]
void RayGen()
{
    PERF_COUNTER(rayGenCount, 1);

    RayPayload payload;
    payload.color = float3(0, 0, 0);

    for (uint s = 0; s < AA_SAMPLES; s++)
    {
        payload.sampleIndex = s;

        float3 origin, direction;
        GenerateCameraRay(
            DispatchRaysDimensions().xy,
            DispatchRaysIndex().xy + AA_SAMPLE_OFFSET_TABLE[s] * float2(1, -1), // Y direction is flipped vs beam vis shader
            origin,
            direction);

        RayDesc rayDesc =
        {
            origin,
            0.0f,
            direction,
            FLT_MAX,
        };

        TraceRay(
            g_accel,
            RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0,
            HIT_GROUP_PRIMARY, HIT_GROUP_COUNT, HIT_GROUP_PRIMARY,
            rayDesc, payload);
    }

    g_screenOutput[DispatchRaysIndex().xy] = float4(payload.color / AA_SAMPLES, 1);
}

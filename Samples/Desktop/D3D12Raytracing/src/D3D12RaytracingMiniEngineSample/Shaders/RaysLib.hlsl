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

[shader("closesthit")]
void Hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
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
    GenerateCameraRay(DispatchRaysDimensions().xy, DispatchRaysIndex().xy + uint2(1, 0), rayOriginDX, rayDirDX);
    GenerateCameraRay(DispatchRaysDimensions().xy, DispatchRaysIndex().xy + uint2(0, 1), rayOriginDY, rayDirDY);

    Triangle triPos = triFetch(meshID, primID);
    float4 uvwtDX = triIntersectNoFail(rayOriginDX, rayDirDX, triPos);
    float4 uvwtDY = triIntersectNoFail(rayOriginDY, rayDirDY, triPos);

    // hopefully the optimizer doesn't completely give up on life at this point, this should just be some extra math
    // reusing the stuff we loaded in the first triFetchAndInterpolate
    TriInterpolated triDX = triFetchAndInterpolate(meshID, primID, uvwtDX.xyz);
    TriInterpolated triDY = triFetchAndInterpolate(meshID, primID, uvwtDY.xyz);

    float2 uvDx = triDX.uv - tri.uv;
    float2 uvDy = triDY.uv - tri.uv;

    float3 diffuseColor = g_localTexture.SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb;

    float3 normal = g_localNormal.SampleGrad(g_s0, tri.uv, uvDx, uvDy).rgb * 2 - 1;
    float gloss = 128;
    AntiAliasSpecular(normal, gloss);

    float3x3 tbn = float3x3(
        normalize(tri.tangent),
        normalize(tri.bitangent),
        normalize(tri.normal));
    normal = normalize(mul(normal, tbn));

    float3 viewDir = normalize(-rayDir);
    float specularMask = .1; // TODO: read the texture

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

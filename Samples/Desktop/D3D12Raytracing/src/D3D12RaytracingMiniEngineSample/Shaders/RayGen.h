#pragma once

#include "RayCommon.h"

inline void GenerateCameraRay(
    uint2 pixelDim,
    uint2 pixelPos,
    out float3 origin,
    out float3 direction)
{
    float2 xy = pixelPos + 0.5; // center in the middle of the pixel
    float2 screenPos = xy / float2(pixelDim) * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    origin = dynamicConstants.worldCameraPosition;
    direction = mul((float3x3)dynamicConstants.cameraToWorld, float3(screenPos, -1));
}

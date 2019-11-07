#pragma once

#include "RayCommon.h"

// https://msdn.microsoft.com/en-us/library/windows/desktop/Ff476218(v=VS.85).aspx
static const float2 sampleOffset1x[1] = {
	{(1.0 / 16.0 * 0), (1.0 / 16.0 * 0)},
};
static const float2 sampleOffset2x[2] = {
	{(1.0 / 16.0 * 4), (1.0 / 16.0 * 4)},
	{(1.0 / 16.0 * -4), (1.0 / 16.0 * -4)},
};
static const float2 sampleOffset4x[4] = {
	{(1.0 / 16.0 * -2), (1.0 / 16.0 * -6)},
	{(1.0 / 16.0 * 6), (1.0 / 16.0 * -2)},
	{(1.0 / 16.0 * -6), (1.0 / 16.0 * 2)},
	{(1.0 / 16.0 * 2), (1.0 / 16.0 * 6)},
};
static const float2 sampleOffset8x[8] = {
	{(1.0 / 16.0 * 1), (1.0 / 16.0 * -3)},
	{(1.0 / 16.0 * -1), (1.0 / 16.0 * 3)},
	{(1.0 / 16.0 * 5), (1.0 / 16.0 * 1)},
	{(1.0 / 16.0 * -3), (1.0 / 16.0 * -5)},
	{(1.0 / 16.0 * -5), (1.0 / 16.0 * 5)},
	{(1.0 / 16.0 * -7), (1.0 / 16.0 * -1)},
	{(1.0 / 16.0 * 3), (1.0 / 16.0 * 7)},
	{(1.0 / 16.0 * 7), (1.0 / 16.0 * -7)},
};
static const float2 sampleOffset16x[16] = {
	{(1.0 / 16.0 * 1), (1.0 / 16.0 * 1)},
	{(1.0 / 16.0 * -1), (1.0 / 16.0 * -3)},
	{(1.0 / 16.0 * -3), (1.0 / 16.0 * 2)},
	{(1.0 / 16.0 * 4), (1.0 / 16.0 * -1)},
	{(1.0 / 16.0 * -5), (1.0 / 16.0 * -2)},
	{(1.0 / 16.0 * 2), (1.0 / 16.0 * 5)},
	{(1.0 / 16.0 * 5), (1.0 / 16.0 * 3)},
	{(1.0 / 16.0 * 3), (1.0 / 16.0 * -5)},
	{(1.0 / 16.0 * -2), (1.0 / 16.0 * 6)},
	{(1.0 / 16.0 * 0), (1.0 / 16.0 * -7)},
	{(1.0 / 16.0 * -4), (1.0 / 16.0 * -6)},
	{(1.0 / 16.0 * -6), (1.0 / 16.0 * 4)},
	{(1.0 / 16.0 * -8), (1.0 / 16.0 * 0)},
	{(1.0 / 16.0 * 7), (1.0 / 16.0 * -4)},
	{(1.0 / 16.0 * 6), (1.0 / 16.0 * 7)},
	{(1.0 / 16.0 * -7), (1.0 / 16.0 * -8)},
};

inline void GenerateCameraRay(
    uint2 pixelDim,
    float2 pixelPos,
    out float3 origin,
    out float3 direction)
{
    float2 xy = pixelPos + .5f;
    float2 screenPos = xy / float2(pixelDim) * 2.0 - 1.0;

    screenPos -= float2(dynamicConstants.jitterNormalizedX, dynamicConstants.jitterNormalizedY);

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    origin = dynamicConstants.worldCameraPosition;
    direction = mul((float3x3)dynamicConstants.cameraToWorld, float3(screenPos, -1));
}

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

void GenerateCameraRay(
    uint2 pixelDim,
    float2 pixelPos,
    out float3 origin,
    out float3 dir)
{
    origin = dynamicConstants.worldCameraPosition;

    float2 xy = pixelPos + .5f;
    float2 screenPos = xy / float2(pixelDim) * 2.0 - 1.0;

    screenPos -= float2(dynamicConstants.jitterNormalizedX, dynamicConstants.jitterNormalizedY);

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    dir = mul((float3x3)dynamicConstants.cameraToWorld, float3(screenPos, -1));
}

// note - outputs in the correct winding order for creating a Frustum
void GenerateTileRays(
    uint2 tileDim,
    uint2 tilePos,
    out float3 origin,
    out float3 dir[4])
{
    origin = dynamicConstants.worldCameraPosition;

    float2 scale = 2.0f / float2(tileDim);
    float2 bias = -float2(dynamicConstants.jitterNormalizedX, dynamicConstants.jitterNormalizedY) - 1.0f;
    float2 screenPos00 = (tilePos + uint2(0, 1)) * scale + bias;
    float2 screenPos10 = (tilePos + uint2(1, 1)) * scale + bias;
    float2 screenPos11 = (tilePos + uint2(1, 0)) * scale + bias;
    float2 screenPos01 = (tilePos + uint2(0, 0)) * scale + bias;

    // Invert Y for DirectX-style coordinates
    screenPos00.y = -screenPos00.y;
    screenPos10.y = -screenPos10.y;
    screenPos11.y = -screenPos11.y;
    screenPos01.y = -screenPos01.y;

    float3x3 rotation = (float3x3)dynamicConstants.cameraToWorld;
    dir[0] = mul(rotation, float3(screenPos00, -1));
    dir[1] = mul(rotation, float3(screenPos10, -1));
    dir[2] = mul(rotation, float3(screenPos11, -1));
    dir[3] = mul(rotation, float3(screenPos01, -1));
}

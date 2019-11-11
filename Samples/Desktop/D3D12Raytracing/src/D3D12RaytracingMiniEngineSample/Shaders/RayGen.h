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

    float2 scale = 2.0f / float2(pixelDim);
    float2 bias = -float2(dynamicConstants.jitterNormalizedX, dynamicConstants.jitterNormalizedY) - 1.0f;
    float2 screenPos = (pixelPos + .5f) * scale + bias; // pixel center

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    float3x3 rotation = (float3x3)dynamicConstants.cameraToWorld;
    dir = mul(rotation, float3(screenPos, -1));
}

// for a simple 2D grid projection with square pixels, there's not much to this
void GenerateCameraRayFootprint(
    uint2 pixelDim,
    out float3 majorDirDiff,
    out float3 minorDirDiff)
{
    float3 major = float3(-2.0f / pixelDim.x, 0, 0);
    float3 minor = float3(0, -2.0f / pixelDim.y, 0); // flip Y for DX Y convention

    float3x3 rotation = (float3x3)dynamicConstants.cameraToWorld;
    majorDirDiff = mul(rotation, major);
    minorDirDiff = mul(rotation, minor);
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
    // Y delta is flipped due to DX Y convention, see below
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

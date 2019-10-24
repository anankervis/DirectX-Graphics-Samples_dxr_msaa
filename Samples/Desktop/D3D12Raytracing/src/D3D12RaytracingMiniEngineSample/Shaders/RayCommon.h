#pragma once

#ifndef HLSL
# include "HlslCompat.h"
#endif

#define QUAD_READ_GROUPSHARED_FALLBACK 1
// Note that the AABBs are enlarged to be conservative from the original camera viewpoint,
// and this isn't updated as you move the camera around.
#define EMULATE_CONSERVATIVE_BEAMS_VIA_AABB_ENLARGEMENT 1

#define AA_SAMPLES 16
#define AA_SAMPLE_OFFSET_TABLE sampleOffset16x

#define TILE_DIM_LOG2 3 // 8x8
#define TILE_DIM (1 << TILE_DIM_LOG2)
#define TILE_SIZE (TILE_DIM * TILE_DIM)
#define TILE_MAX_TRIS 512

struct TileTri
{
    uint id[TILE_MAX_TRIS]; // material + primitive IDs
};

struct RayTraceMeshInfo
{
    uint indexOffset;
    uint attrOffsetTexcoord0;
    uint attrOffsetNormal;
    uint attrOffsetTangent;
    uint attrOffsetBitangent;
    uint attrOffsetPos;
    uint attrStride;
    uint materialID;
};

// Volatile part (can be split into its own CBV). 
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3 worldCameraPosition;

    uint tilesX;
    uint tilesY;
};

struct RootConstants
{
    uint meshID;
};

struct RayPayload
{
    float3 color;
};

struct BeamPayload
{
    uint pad;
};
struct BeamHitAttribs
{
    uint pad;
};

#ifdef HLSL

# ifndef SINGLE
static const float FLT_MAX = asfloat(0x7F7FFFFF);
# endif

RaytracingAccelerationStructure g_accel : register(t0);

StructuredBuffer<RayTraceMeshInfo> g_meshInfo : register(t1);
ByteAddressBuffer g_indices : register(t2);
ByteAddressBuffer g_attributes : register(t3);

Texture2D<float4> g_localTexture : register(t10);
Texture2D<float4> g_localNormal : register(t11);

SamplerState      g_s0 : register(s0);

RWTexture2D<float4> g_screenOutput : register(u2);
RWStructuredBuffer<uint> g_tileTriCounts : register(u3);
RWStructuredBuffer<TileTri> g_tileTris : register(u4);

cbuffer b1 : register(b1)
{
    DynamicCB dynamicConstants;
};

cbuffer b3 : register(b3)
{
    RootConstants rootConstants;
};

#endif

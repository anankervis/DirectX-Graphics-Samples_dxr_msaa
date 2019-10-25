#pragma once

#ifndef HLSL
# include "HlslCompat.h"
#endif

#define QUAD_READ_GROUPSHARED_FALLBACK 1
// Note that the AABBs are enlarged to be conservative from the original camera viewpoint,
// and this isn't updated as you move the camera around.
#define EMULATE_CONSERVATIVE_BEAMS_VIA_AABB_ENLARGEMENT 1

// 0 = 1x, 1 = 2x, 2 = 4x, 3 = 8x, 4 = 16x
// Don't forget to update AA_SAMPLE_OFFSET_TABLE to point to the corresponding table.
#define AA_SAMPLES_LOG2 3
#define AA_SAMPLE_OFFSET_TABLE sampleOffset8x
#define AA_SAMPLES (1 << AA_SAMPLES_LOG2)
#define AA_SAMPLE_MASK ((uint(1) << AA_SAMPLES) - 1)

// 8x4 tiles
#define TILE_DIM_LOG2_X 3
#define TILE_DIM_LOG2_Y 2
#define TILE_DIM_X (1 << TILE_DIM_LOG2_X)
#define TILE_DIM_Y (1 << TILE_DIM_LOG2_Y)
#define TILE_SIZE (TILE_DIM_X * TILE_DIM_Y)
#define TILE_MAX_TRIS 512

#define QUAD_DIM_X 2
#define QUAD_DIM_Y 2
#define QUAD_SIZE (QUAD_DIM_X * QUAD_DIM_Y)
#define QUADS_PER_TILE_X (TILE_DIM_X / QUAD_DIM_X)
#define QUADS_PER_TILE_Y (TILE_DIM_Y / QUAD_DIM_Y)
#define QUADS_PER_TILE (QUADS_PER_TILE_X * QUADS_PER_TILE_Y)

#define MAX_TRIS_PER_QUAD (QUAD_SIZE * AA_SAMPLES)
#define MAX_SHADE_QUADS_PER_TILE (MAX_TRIS_PER_QUAD * QUADS_PER_TILE)

#define BAD_TRI_ID (~uint(0))

struct TileTri
{
    uint id[TILE_MAX_TRIS]; // mesh + primitive IDs
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

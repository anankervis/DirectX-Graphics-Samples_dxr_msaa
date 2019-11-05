#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "RayGen.h"
#include "Shading.h"
#include "TriFetch.h"

#pragma warning (disable: 3078) // this doesn't seem to work with the new HLSL compiler...

struct ShadePixel
{
    uint id; // mesh + primitive IDs
    uint sampleMask;
};

#define SORT_SIZE AA_SAMPLES
#define SORT_T uint
#define SORT_CMP_LESS(a, b) (a < b)
#include "Sort.h"

groupshared uint tileQuadCount;

#if QUAD_READ_GROUPSHARED_FALLBACK
groupshared uint qr_uint[TILE_SIZE];
#endif

void EmitQuad(
    uint tileIndex,
    uint threadID, uint quadIndex, uint quadLocalIndex,
    uint id, uint localMatchCount, bool quadDone)
{
#if QUAD_READ_GROUPSHARED_FALLBACK
    qr_uint[threadID] = localMatchCount;
    GroupMemoryBarrierWithGroupSync();
// TODO: only the first quad thread needs to execute here
    uint threadID00 = threadID & ~(QUAD_SIZE - 1);
    uint matchCount00 = qr_uint[threadID00 + 0];
    uint matchCount10 = qr_uint[threadID00 + 1];
    uint matchCount01 = qr_uint[threadID00 + 2];
    uint matchCount11 = qr_uint[threadID00 + 3];
    GroupMemoryBarrierWithGroupSync();
#else
    uint matchCount00 = QuadReadLaneAt(localID, 0);
    uint matchCount10 = QuadReadLaneAt(localID, 1);
    uint matchCount01 = QuadReadLaneAt(localID, 2);
    uint matchCount11 = QuadReadLaneAt(localID, 3);
#endif

    ShadeQuad shadeQuad;
    shadeQuad.id = id;
    shadeQuad.bits = quadIndex;
    if (quadDone)
        shadeQuad.bits |= 1 << (QUADS_PER_TILE_LOG2 + 0);
    if (id != BAD_TRI_ID)
    {
        shadeQuad.bits |= matchCount00 << (QUADS_PER_TILE_LOG2 + 1 + (AA_SAMPLES_LOG2 + 1) * 0);
        shadeQuad.bits |= matchCount10 << (QUADS_PER_TILE_LOG2 + 1 + (AA_SAMPLES_LOG2 + 1) * 1);
        shadeQuad.bits |= matchCount01 << (QUADS_PER_TILE_LOG2 + 1 + (AA_SAMPLES_LOG2 + 1) * 2);
        shadeQuad.bits |= matchCount11 << (QUADS_PER_TILE_LOG2 + 1 + (AA_SAMPLES_LOG2 + 1) * 3);
    }

    if (quadLocalIndex == 0)
    {
        uint outputSlot;
        InterlockedAdd(tileQuadCount, 1, outputSlot);
        if (outputSlot < MAX_SHADE_QUADS_PER_TILE)
        {
            g_tileShadeQuads[tileIndex].quads[outputSlot] = shadeQuad;
        }
    }
}

[numthreads(TILE_SIZE, 1, 1)]
[RootSignature(
    "CBV(b0),"
    "CBV(b1),"
    "DescriptorTable(UAV(u2, numDescriptors = 5)),"
    "DescriptorTable(SRV(t1, numDescriptors = 3)),"
    "DescriptorTable(SRV(t100, numDescriptors = unbounded)),"
    "StaticSampler(s0, maxAnisotropy = 8),"
)]
void BeamsQuadVis(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileX = groupID.x;
    uint tileY = groupID.y;
    uint tileIndex = tileY * dynamicConstants.tilesX + tileX;
    uint threadID = groupThreadID.x;
    uint quadLocalIndex = threadID & (QUAD_SIZE - 1);
    uint quadIndex = threadID / QUAD_SIZE;
    uint quadLaneMask = (QUAD_SIZE - 1) << (quadIndex * QUAD_SIZE);
    uint localX;
    uint localY;
    threadIndexToQuadSwizzle(threadID, localX, localY);
    uint pixelX = tileX * TILE_DIM_X + localX;
    uint pixelY = tileY * TILE_DIM_Y + localY;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM_X;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM_Y;

    uint tileTriCount = g_tileTriCounts[tileIndex];
    if (tileTriCount <= 0)
    {
        // no triangles overlap this tile
        g_tileShadeQuadsCount[tileIndex] = 0;
        return;
    }
    else if (tileTriCount > TILE_MAX_TRIS)
    {
        // tile tri list overflowed
        g_tileShadeQuadsCount[tileIndex] = ~uint(0);
        return;
    }

    float nearestT[AA_SAMPLES];
    uint nearestID[AA_SAMPLES];
    {for (uint s = 0; s < AA_SAMPLES; s++)
    {
        nearestT[s] = FLT_MAX;
        nearestID[s] = BAD_TRI_ID;
    }}
    for (uint tileTriIndex = 0; tileTriIndex < tileTriCount; tileTriIndex++)
    {
        uint id = g_tileTris[tileIndex].id[tileTriIndex];
        uint meshID = id >> 16;
        uint primID = id & 0xffff;

        Triangle tri = triFetch(meshID, primID);
        
        for (uint s = 0; s < AA_SAMPLES; s++)
        {
            float3 rayOrigin;
            float3 rayDir;
            GenerateCameraRay(
                uint2(pixelDimX, pixelDimY),
                float2(pixelX, pixelY) + AA_SAMPLE_OFFSET_TABLE[s],
                rayOrigin, rayDir);

            float4 uvwt = triIntersect(rayOrigin, rayDir, tri);

            if (uvwt.x >= 0 && uvwt.y >= 0 && uvwt.z >= 0 &&
                uvwt.w < nearestT[s])
            {
                nearestT[s] = uvwt.w;
                nearestID[s] = id;
            }
        }
    }

    // Beware packing bits into the sort key and/or sign-extending it on unpack like HVVR does...
    // HLSL likes to silently convert uint to int (for example, the min intrinsic).
    sortBitonic(nearestID);
// TODO: move sort result into groupshared for indexing code gen?

    if (threadID == 0)
        tileQuadCount = 0;
    GroupMemoryBarrierWithGroupSync();

    uint matchID = BAD_TRI_ID;
    uint localS = 0;
    uint localMatchCount = 0;
    while (true)
    {
        bool localDone = localS >= AA_SAMPLES;
        bool quadDone = (WaveActiveBallot(localDone).x & quadLaneMask) == quadLaneMask;

        uint localID = BAD_TRI_ID;
        if (!localDone)
            localID = nearestID[localS];

#if QUAD_READ_GROUPSHARED_FALLBACK
        // groupshared fallback
        qr_uint[threadID] = localID;
        GroupMemoryBarrierWithGroupSync();
        uint threadID00 = threadID & ~(QUAD_SIZE - 1);
        uint id00 = qr_uint[threadID00 + 0];
        uint id10 = qr_uint[threadID00 + 1];
        uint id01 = qr_uint[threadID00 + 2];
        uint id11 = qr_uint[threadID00 + 3];
        GroupMemoryBarrierWithGroupSync();
#else
        uint id00 = QuadReadLaneAt(localID, 0);
        uint id10 = QuadReadLaneAt(localID, 1);
        uint id01 = QuadReadLaneAt(localID, 2);
        uint id11 = QuadReadLaneAt(localID, 3);
#endif
        uint minID = min(min(id00, id10), min(id01, id11));

        if (minID != matchID || quadDone)
        {
            // emit the current quad
            if (matchID != BAD_TRI_ID ||    // don't emit the first placeholder BAD_TRI_ID quad...
                quadDone)                   // ...unless we reached the end without any valid hits
            {
                EmitQuad(
                    tileIndex,
                    threadID, quadIndex, quadLocalIndex,
                    matchID, localMatchCount, quadDone);
            }

            // start a new quad
            matchID = minID;
            localMatchCount = 0;
        }

        if (quadDone)
            break;

        bool localMatch = (localID == matchID);
        if (localMatch)
        {
            localMatchCount++;
            localS++;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    g_tileShadeQuadsCount[tileIndex] = tileQuadCount;
}

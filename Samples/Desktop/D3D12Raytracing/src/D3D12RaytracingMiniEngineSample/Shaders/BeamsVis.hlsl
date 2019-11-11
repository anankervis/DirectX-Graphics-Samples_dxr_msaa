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

#define TRI_CACHE_SIZE WAVE_SIZE

struct TriCacheEntry
{
    TriTile triTile;
    uint id;
};
groupshared TriCacheEntry triCache[TRI_CACHE_SIZE];
groupshared uint triCacheCount;

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
        PERF_COUNTER(visShadeQuads, 1);

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
    "DescriptorTable(UAV(u2, numDescriptors = 6)),"
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
    uint laneID = threadID % WAVE_SIZE;
    uint laneMaskLT = (1 << laneID) - 1;
    uint quadLocalIndex = threadID & (QUAD_SIZE - 1);
    uint quadIndex = threadID / QUAD_SIZE;
    uint quadLaneMask = uint((1 << QUAD_SIZE) - 1) << (quadIndex * QUAD_SIZE);
    uint localX;
    uint localY;
    threadIndexToQuadSwizzle(threadID, localX, localY);
    uint pixelX = tileX * TILE_DIM_X + localX;
    uint pixelY = tileY * TILE_DIM_Y + localY;
    uint pixelDimX = dynamicConstants.tilesX * TILE_DIM_X;
    uint pixelDimY = dynamicConstants.tilesY * TILE_DIM_Y;

    if (threadID == 0) PERF_COUNTER(visTiles, 1);

    uint tileLeafCount = g_tileLeafCounts[tileIndex];
    if (tileLeafCount <= 0)
    {
        // no leaves overlap this tile
        if (threadID == 0) PERF_COUNTER(visTileNoLeaves, 1);
        g_tileShadeQuadsCount[tileIndex] = 0;
        return;
    }
    else if (tileLeafCount > TILE_MAX_LEAVES)
    {
        // tile leaf list overflowed
        if (threadID == 0) PERF_COUNTER(visTileOverflow, 1);
        g_tileShadeQuadsCount[tileIndex] = ~uint(0);
        return;
    }

    // TODO: some of this work could probably be precomputed
    float3 tileOrigin;
    float3 tileDirs[4];
    GenerateTileRays(uint2(dynamicConstants.tilesX, dynamicConstants.tilesY), uint2(tileX, tileY), tileOrigin, tileDirs);
    Frustum tileFrustum = FrustumCreate(tileOrigin, tileDirs);

    if (threadID == 0)
    {
        triCacheCount = 0;
        tileQuadCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    float nearestT[AA_SAMPLES];
// TODO: nearestID can be moved to groupshared to reduce register pressure
// (but should be moved back to registers for the sort)
    uint nearestID[AA_SAMPLES];
    {for (uint s = 0; s < AA_SAMPLES; s++)
    {
        nearestT[s] = FLT_MAX;
        nearestID[s] = BAD_TRI_ID;
    }}

    // threads cooperate to fetch and coarse cull the triangles
// TODO: one tri per thread, instead of one leaf per thread, once we get TRIS_PER_AABB > 1
    uint fetchIterations = (tileLeafCount + TILE_SIZE - 1) / TILE_SIZE;
    for (uint f = 0; f < fetchIterations; f++)
    {
        if (threadID == 0) PERF_COUNTER(visTileFetchIterations, 1);

        uint tileLeafIndex = f * TILE_SIZE + threadID;

        if (tileLeafIndex < tileLeafCount)
        {
            PERF_COUNTER(visTileLeaves, 1);

            uint aabbID = g_tileLeaves[tileIndex].id[tileLeafIndex];
            uint meshID = aabbID >> PRIM_ID_BITS;
            uint primID = aabbID & PRIM_ID_MASK;

            uint meshTriCount = g_meshInfo[meshID].triCount;
            for (uint triID = primID * TRIS_PER_AABB; triID < (primID + 1) * TRIS_PER_AABB; triID++)
            {
                // TODO: remove this in favor of inserting a duplicate or degenerate triangle
                if (triID >= meshTriCount)
                    break;

                PERF_COUNTER(visTileTrisIn, 1);

                Triangle tri = triFetch(meshID, triID);

                bool outputTri = false;
                TriTile triTile;

                // test the triangle against the tile frustum's planes
                if (FrustumTest(tileFrustum, tri))
                {
                    // test for backfacing and intersection before ray origin
                    if (TriTileSetup(tri, tileOrigin, triTile))
                    {
                        // test UVW overlap
                        if (FrustumTestUVW(tileOrigin, tileDirs, tri))
                        {
                            outputTri = true;
                        }
                        else
                        {
                            PERF_COUNTER(visTileTrisCulledTileUVW, 1);
                        }
                    }
                    else
                    {
                        PERF_COUNTER(visTileTrisCulledTileSetup, 1);
                    }
                }
                else
                {
                    PERF_COUNTER(visTileTrisCulledTileFrustum, 1);
                }

                uint appendMask = WaveActiveBallot(outputTri).x;
                uint appendCount = countbits(appendMask);
                uint appendSlotBase;
                if (laneID == 0)
                    InterlockedAdd(triCacheCount, appendCount, appendSlotBase);
                appendSlotBase = WaveReadLaneAt(appendSlotBase, 0);
                uint appendSlot = appendSlotBase + countbits(appendMask & laneMaskLT);

                if (outputTri)
                {
                    triCache[appendSlot].triTile = triTile;
                    triCache[appendSlot].id = (meshID << PRIM_ID_BITS) | triID;

                    PERF_COUNTER(visTileTrisPass, 1);
                }
            }
        }

// TODO: at this point, we could check for cache capacity and use a "continue" statement to keep accumulating

        GroupMemoryBarrierWithGroupSync();

        // process the surviving triangles in the cache
        for (uint cacheIndex = 0; cacheIndex < triCacheCount; cacheIndex++)
        {
            TriTile triTile = triCache[cacheIndex].triTile;
            uint id = triCache[cacheIndex].id;

            float3 rayOriginCenter;
            float3 rayDirCenter;
            GenerateCameraRay(
                uint2(pixelDimX, pixelDimY),
                float2(pixelX, pixelY),
                rayOriginCenter, rayDirCenter);

            float3 majorDirDiff;
            float3 minorDirDiff;
            GenerateCameraRayFootprint(
                uint2(pixelDimX, pixelDimY),
                majorDirDiff, minorDirDiff);

            TriThread triThread = TriThreadSetup(triTile, rayDirCenter, majorDirDiff, minorDirDiff);

            for (uint s = 0; s < AA_SAMPLES; s++)
            {
                if (TriThreadTest(triTile, triThread, AA_SAMPLE_OFFSET_TABLE[s], nearestT[s]))
                {
                    nearestID[s] = id;
                }
            }
        }

        // reset the cache
        if (threadID == 0)
            triCacheCount = 0;
        GroupMemoryBarrierWithGroupSync();
    }

    // Beware packing bits into the sort key and/or sign-extending it on unpack like HVVR does...
    // HLSL likes to silently convert uint to int (for example, the min intrinsic).
    sortBitonic(nearestID);
// TODO: move sort result into groupshared for indexing code gen?

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
// TODO: remove quadDone and the BAD_TRI_ID special quad case, now that we have a shade quad count
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

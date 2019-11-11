#define HLSL

#include "Intersect.h"
#include "RayCommon.h"
#include "RayGen.h"
#include "TriFetch.h"

// Record the triangle and move on. We'll compute coverage later.
[shader("anyhit")]
void AnyHit(inout BeamPayload payload, in BeamHitAttribs attr)
{
    PERF_COUNTER(anyHitCount, 1);

    uint tileX = DispatchRaysIndex().x;
    uint tileY = DispatchRaysIndex().y;
    uint tileIndex = tileY * DispatchRaysDimensions().x + tileX;

    uint meshID = rootConstants.meshID;
    uint triID = PrimitiveIndex();

    uint triSlot;
    InterlockedAdd(g_tileTriCounts[tileIndex], 1, triSlot);

    if (triSlot >= TILE_MAX_TRIS)
        return;

    uint id = (meshID << PRIM_ID_BITS) | triID;

    g_tileTris[tileIndex].id[triSlot] = id;
}

[shader("intersection")]
void Intersection()
{
    PERF_COUNTER(intersectCount, 1);

    float tHitAABB = RayTCurrent();

    BeamHitAttribs attr;

    {
        uint tileX = DispatchRaysIndex().x;
        uint tileY = DispatchRaysIndex().y;

        uint meshID = rootConstants.meshID;
        uint primID = PrimitiveIndex();

// TODO: this is only needed if TRIS_PER_AABB > 1
        uint meshTriCount = g_meshInfo[meshID].triCount;

        // TODO: some of this work could probably be precomputed
        float3 tileOrigin;
        float3 tileDirs[4];
        GenerateTileRays(uint2(dynamicConstants.tilesX, dynamicConstants.tilesY), uint2(tileX, tileY), tileOrigin, tileDirs);
        Frustum tileFrustum = FrustumCreate(tileOrigin, tileDirs);

        bool outputLeaf = false;
        for (uint triID = primID * TRIS_PER_AABB; triID < (primID + 1) * TRIS_PER_AABB; triID++)
        {
            TriTile triTile;

            if (triID < meshTriCount)
            {
                PERF_COUNTER(intersectTrisIn, 1);
                Triangle tri = triFetch(meshID, triID);

                // test the triangle against the tile frustum's planes
                if (FrustumTest(tileFrustum, tri))
                {
                    // test for backfacing and intersection before ray origin
                    if (TriTileSetup(tri, tileOrigin, triTile))
                    {
                        // test UVW interval overlap
                        float conservativeMaxT;
                        bool partialCoverage;
                        bool fullCoverage;
                        FrustumTest_ConservativeT(
                            tileOrigin, tileDirs, tri,
                            conservativeMaxT, partialCoverage, fullCoverage);

                        if (fullCoverage)
                        {
                            ReportHit(conservativeMaxT, 0, attr);
                            PERF_COUNTER(intersectTrisFullCoverage, 1);
                        }
                        else if (partialCoverage)
                        {
                            ReportHit(tHitAABB, 0, attr);
                            PERF_COUNTER(intersectTrisPartialCoverage, 1);
                        }
                        else
                        {
                            PERF_COUNTER(intersectTrisCulledTileUVW, 1);
                        }
                    }
                    else
                    {
                        PERF_COUNTER(intersectTrisCulledTileSetup, 1);
                    }
                }
                else
                {
                    PERF_COUNTER(intersectTrisCulledTileFrustum, 1);
                }
            }
        }
    }
}

[shader("miss")]
void Miss(inout BeamPayload payload)
{
    PERF_COUNTER(missCount, 1);
}

[shader("raygeneration")]
void RayGen()
{
    PERF_COUNTER(rayGenCount, 1);

    float3 origin, direction;
    GenerateCameraRay(DispatchRaysDimensions().xy, DispatchRaysIndex().xy, origin, direction);

    RayDesc rayDesc =
    {
        origin,
        0.0f,
        direction,
        FLT_MAX
    };

    BeamPayload payload;

    TraceRay(g_accel, RAY_FLAG_NONE, ~0, 0, 1, 0, rayDesc, payload);
}

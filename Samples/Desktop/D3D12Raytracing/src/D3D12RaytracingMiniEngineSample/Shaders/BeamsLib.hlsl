#define HLSL

#include "RayCommon.h"
#include "RayGen.h"

struct HitAttribs
{
};

/*
Record the triangle and move on. We'll compute coverage later.

Yes, it would be nice to have per-sample coverage auto-generated and fed into this function...
...but how would that work with beams?

Single-pixel rays w/ MSAA can be treated as a collection of 2, 4, 8, 16 subrays w/ coverage mask.
Beams represent the conservative volume, which may be a collection of pixels or a spatial query.
*/
[shader("anyhit")]
void AnyHit(inout BeamPayload payload, in HitAttribs attr)
{
    uint tileX = DispatchRaysIndex().x;
    uint tileY = DispatchRaysIndex().y;
    uint tileIndex = tileY * DispatchRaysDimensions().x + tileX;

    uint triSlot;
    InterlockedAdd(g_tileTriCounts[tileIndex], 1, triSlot);

    if (triSlot >= TILE_MAX_TRIS)
        return;

    uint id = (rootConstants.meshID << 16) | PrimitiveIndex();

    g_tileTris[tileIndex].id[triSlot] = id;
}

[shader("intersection")]
void Intersection()
{
    float tHitAABB = RayTCurrent();

    HitAttribs attr;

    ReportHit(tHitAABB, 0, attr);
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

    BeamPayload payload;

    TraceRay(g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
}

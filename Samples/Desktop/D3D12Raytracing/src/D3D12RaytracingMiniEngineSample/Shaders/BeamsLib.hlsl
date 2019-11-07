#define HLSL

#include "RayCommon.h"
#include "RayGen.h"

// Record the triangle and move on. We'll compute coverage later.
[shader("anyhit")]
void AnyHit(inout BeamPayload payload, in BeamHitAttribs attr)
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

    BeamHitAttribs attr;

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

#define HLSL

#include "RayCommon.h"
#include "RayGen.h"

// Record the leaf node and move on. We'll compute coverage later.
[shader("anyhit")]
void AnyHit(inout BeamPayload payload, in BeamHitAttribs attr)
{
    uint tileX = DispatchRaysIndex().x;
    uint tileY = DispatchRaysIndex().y;
    uint tileIndex = tileY * DispatchRaysDimensions().x + tileX;

    uint leafSlot;
    InterlockedAdd(g_tileLeafCounts[tileIndex], 1, leafSlot);

    if (leafSlot >= TILE_MAX_LEAVES)
        return;

    uint id = (rootConstants.meshID << PRIM_ID_BITS) | PrimitiveIndex();

    g_tileLeaves[tileIndex].id[leafSlot] = id;
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

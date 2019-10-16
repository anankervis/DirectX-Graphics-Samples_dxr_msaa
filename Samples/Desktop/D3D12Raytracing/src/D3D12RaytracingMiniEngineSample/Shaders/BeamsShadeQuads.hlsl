#define HLSL

#include "RayCommon.h"
#include "Shading.h"

/*
Shade the quads
*/
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
[RootSignature(
    "DescriptorTable(UAV(u2, numDescriptors = 3)),"
    "CBV(b1),"
)]
void ShadeQuads(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileX = groupID.x;
    uint tileY = groupID.y;
    uint tileIndex = tileY * dynamicConstants.tilesX + tileX;
    //uint tileIndex = tileY * 960 + tileX;

    uint triCount = g_tileTriCounts[tileIndex];
    if (triCount <= 0)
    {
        g_screenOutput[dispatchThreadID.xy] = float4(0, 0, 1, 1);
        return;
    }
    else if (triCount > TILE_MAX_TRIS)
    {
        g_screenOutput[dispatchThreadID.xy] = float4(1, 0, 0, 1);
        return;
    }

    g_screenOutput[dispatchThreadID.xy] = float4(0, triCount * (1.0f / TILE_MAX_TRIS), 0, 1);
}

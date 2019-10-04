#define HLSL

#include "RayCommon.h"
#include "Shading.h"

/*
Shade the quads
*/
[numthreads(BEAM_SIZE, BEAM_SIZE, 1)]
[RootSignature(
    "DescriptorTable(UAV(u2)),"
)]
void ShadeQuads(
    uint groupIndex : SV_GroupIndex,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    g_screenOutput[dispatchThreadID.xy] = float4(0, 1, 0, 1);
}

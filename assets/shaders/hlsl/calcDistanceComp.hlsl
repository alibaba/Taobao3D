#include "common.hlsl"

#define CAMERA_NEAR 0.01f
#define CAMERA_FAR 5.0f

StructuredBuffer<uint> _SortParams : register(t0);  // uint[8] type
StructuredBuffer<float> _PointDistances : register(t1);
StructuredBuffer<uint> _SplatSortKeys : register(t2);

RWStructuredBuffer<uint> _SplatSortDistances : register(u3);

uint GetSortSplatCount() { return _SortParams[0]; }

float CalcDistancesSimple(uint idx)
{
    return _PointDistances[_SplatSortKeys[idx]];
}

uint FloatToEncodedUint16(float f)
{
    // sort by uint16 keys
    return clamp((f - CAMERA_NEAR) / (CAMERA_FAR - CAMERA_NEAR), 0.0f, 1.0f) * 0xffff;
}

[numthreads(GROUP_SIZE, 1, 1)]
void CalcUint16Distances(uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    if (idx >= GetSortSplatCount())
        return;

    _SplatSortDistances[idx] = FloatToEncodedUint16(CalcDistancesSimple(idx));
}
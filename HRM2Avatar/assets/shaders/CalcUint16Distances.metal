#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_StructuredBuffer_float
{
    float _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void CalcUint16Distances(const device type_StructuredBuffer_uint& _SortParams [[buffer(0)]], const device type_StructuredBuffer_float& _PointDistances [[buffer(1)]], const device type_StructuredBuffer_uint& _SplatSortKeys [[buffer(2)]], device type_RWStructuredBuffer_uint& _SplatSortDistances [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _SortParams._m0[0u])
        {
            break;
        }
        _SplatSortDistances._m0[gl_GlobalInvocationID.x] = uint(fast::clamp((_PointDistances._m0[_SplatSortKeys._m0[gl_GlobalInvocationID.x]] - 0.00999999977648258209228515625) * 0.20040081441402435302734375, 0.0, 1.0) * 65535.0);
        break;
    } while(false);
}


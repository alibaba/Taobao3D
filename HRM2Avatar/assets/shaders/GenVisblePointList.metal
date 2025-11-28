#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _PointCount;
};

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void GenVisblePointList(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _VisibleMaskPrefixSums [[buffer(1)]], device type_RWStructuredBuffer_uint& _VisiblePointList [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._PointCount.x))
        {
            break;
        }
        uint _109 = _VisibleMaskPrefixSums._m0[gl_GlobalInvocationID.x];
        if (gl_GlobalInvocationID.x == 0u)
        {
            if (_109 == 1u)
            {
                _VisiblePointList._m0[0u] = 0u;
            }
        }
        else
        {
            if (_VisibleMaskPrefixSums._m0[gl_GlobalInvocationID.x - 1u] != _109)
            {
                _VisiblePointList._m0[_109 - 1u] = gl_GlobalInvocationID.x;
            }
        }
        break;
    } while(false);
}


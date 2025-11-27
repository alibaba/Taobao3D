#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void CopyUintBufferIndirect(const device type_StructuredBuffer_uint& _SortParams [[buffer(0)]], const device type_StructuredBuffer_uint& _SrcUintBuffer [[buffer(1)]], device type_RWStructuredBuffer_uint& _DstUintBuffer [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < _SortParams._m0[0u])
    {
        _DstUintBuffer._m0[gl_GlobalInvocationID.x] = _SrcUintBuffer._m0[gl_GlobalInvocationID.x];
    }
}


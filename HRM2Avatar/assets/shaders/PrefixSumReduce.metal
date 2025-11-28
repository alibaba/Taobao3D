#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct type_Constants
{
    float4 _Count;
};

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void PrefixSumReduce(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _ReduceSrcValues [[buffer(1)]], device type_RWStructuredBuffer_uint& _ReduceDstValues [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]])
{
    threadgroup spvUnsafeArray<uint, 1024> gs_TempSums;
    uint _46 = 1024u * gl_WorkGroupID.x;
    uint _48;
    _48 = 0u;
    for (uint _51 = 0u; _51 < 4u; )
    {
        uint _58 = _46 + ((gl_LocalInvocationID.x * 4u) + _51);
        _48 += ((_58 < as_type<uint>(Constants._Count.x)) ? _ReduceSrcValues._m0[_58] : 0u);
        _51++;
        continue;
    }
    uint _67 = simd_sum(_48);
    uint _69 = gl_LocalInvocationID.x / gl_SubgroupSize;
    bool _70 = simd_is_first();
    if (_70)
    {
        gs_TempSums[_69] = _67;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(_69 != 0u))
    {
        uint _84 = simd_sum((gl_LocalInvocationID.x < (256u / gl_SubgroupSize)) ? gs_TempSums[gl_LocalInvocationID.x] : 0u);
        if (gl_LocalInvocationID.x == 0u)
        {
            _ReduceDstValues._m0[gl_WorkGroupID.x] = _84;
        }
    }
}


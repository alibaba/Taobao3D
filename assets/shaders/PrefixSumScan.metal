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

kernel void PrefixSumScan(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]], device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 1024> gs_TempSums;
    threadgroup spvUnsafeArray<uint, 1024> gs_Values;
    uint _51 = as_type<uint>(Constants._Count.x);
    uint _52 = 1024u * gl_WorkGroupID.x;
    for (uint _54 = 0u; _54 < 4u; )
    {
        uint _61 = (gl_LocalInvocationID.x * 4u) + _54;
        uint _62 = _52 + _61;
        gs_Values[_61] = (_62 < _51) ? _ScanSrcValues._m0[_62] : 0u;
        _54++;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _69;
    _69 = 0u;
    for (uint _72 = 0u; _72 < 4u; )
    {
        uint _78 = (gl_LocalInvocationID.x * 4u) + _72;
        uint _80 = gs_Values[_78];
        gs_Values[_78] = _69;
        _69 += _80;
        _72++;
        continue;
    }
    uint _82 = simd_prefix_exclusive_sum(_69);
    uint _84 = gl_LocalInvocationID.x / gl_SubgroupSize;
    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
    {
        gs_TempSums[_84] = _82 + _69;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_LocalInvocationID.x < (256u / gl_SubgroupSize))
    {
        gs_TempSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_TempSums[gl_LocalInvocationID.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _104 = gs_TempSums[_84] + _82;
    for (uint _107 = 0u; _107 < 4u; _107++)
    {
        uint _114 = (gl_LocalInvocationID.x * 4u) + _107;
        uint _115 = _52 + _114;
        if (_115 < _51)
        {
            _ScanDstValues._m0[_115] = gs_Values[_114] + _104;
        }
    }
}


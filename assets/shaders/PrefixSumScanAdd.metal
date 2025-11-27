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

kernel void PrefixSumScanAdd(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]], const device type_StructuredBuffer_uint& _ScanAddSrcValues [[buffer(2)]], device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 1024> gs_TempSums;
    threadgroup spvUnsafeArray<uint, 1024> gs_Values;
    uint _52 = as_type<uint>(Constants._Count.x);
    uint _53 = 1024u * gl_WorkGroupID.x;
    for (uint _55 = 0u; _55 < 4u; )
    {
        uint _62 = (gl_LocalInvocationID.x * 4u) + _55;
        uint _63 = _53 + _62;
        gs_Values[_62] = (_63 < _52) ? _ScanSrcValues._m0[_63] : 0u;
        _55++;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _70;
    _70 = 0u;
    for (uint _73 = 0u; _73 < 4u; )
    {
        uint _79 = (gl_LocalInvocationID.x * 4u) + _73;
        uint _81 = gs_Values[_79];
        gs_Values[_79] = _70;
        _70 += _81;
        _73++;
        continue;
    }
    uint _83 = simd_prefix_exclusive_sum(_70);
    uint _85 = gl_LocalInvocationID.x / gl_SubgroupSize;
    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
    {
        gs_TempSums[_85] = _83 + _70;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_LocalInvocationID.x < (256u / gl_SubgroupSize))
    {
        gs_TempSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_TempSums[gl_LocalInvocationID.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _109 = (gs_TempSums[_85] + _83) + _ScanAddSrcValues._m0[gl_WorkGroupID.x];
    for (uint _112 = 0u; _112 < 4u; _112++)
    {
        uint _119 = (gl_LocalInvocationID.x * 4u) + _112;
        uint _120 = _53 + _119;
        if (_120 < _52)
        {
            _ScanDstValues._m0[_120] = gs_Values[_119] + _109;
        }
    }
}


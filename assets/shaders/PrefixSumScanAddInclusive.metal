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

kernel void PrefixSumScanAddInclusive(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]], const device type_StructuredBuffer_uint& _ScanAddSrcValues [[buffer(2)]], device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 1024> gs_TempSums;
    threadgroup spvUnsafeArray<uint, 1024> gs_Values;
    uint _56 = as_type<uint>(Constants._Count.x);
    uint _57 = 1024u * gl_WorkGroupID.x;
    spvUnsafeArray<uint, 4> _48;
    for (uint _59 = 0u; _59 < 4u; )
    {
        uint _66 = (gl_LocalInvocationID.x * 4u) + _59;
        uint _67 = _57 + _66;
        uint _71 = (_67 < _56) ? _ScanSrcValues._m0[_67] : 0u;
        _48[_59] = _71;
        gs_Values[_66] = _71;
        _59++;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _75;
    _75 = 0u;
    for (uint _78 = 0u; _78 < 4u; )
    {
        uint _84 = (gl_LocalInvocationID.x * 4u) + _78;
        uint _86 = gs_Values[_84];
        gs_Values[_84] = _75;
        _75 += _86;
        _78++;
        continue;
    }
    uint _88 = simd_prefix_exclusive_sum(_75);
    uint _90 = gl_LocalInvocationID.x / gl_SubgroupSize;
    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
    {
        gs_TempSums[_90] = _88 + _75;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_LocalInvocationID.x < (256u / gl_SubgroupSize))
    {
        gs_TempSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_TempSums[gl_LocalInvocationID.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _114 = (gs_TempSums[_90] + _88) + _ScanAddSrcValues._m0[gl_WorkGroupID.x];
    for (uint _117 = 0u; _117 < 4u; _117++)
    {
        uint _124 = (gl_LocalInvocationID.x * 4u) + _117;
        uint _125 = _57 + _124;
        if (_125 < _56)
        {
            _ScanDstValues._m0[_125] = (gs_Values[_124] + _114) + _48[_117];
        }
    }
}


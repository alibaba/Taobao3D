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

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void FfxParallelSortScan(const device type_StructuredBuffer_uint& sort_params [[buffer(0)]], device type_RWStructuredBuffer_uint& rw_scan_source [[buffer(1)]], device type_RWStructuredBuffer_uint& rw_scan_dest [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_LDSSums;
    threadgroup spvUnsafeArray<spvUnsafeArray<uint, 128>, 4> gs_FFX_PARALLELSORT_LDS;
    uint _45 = 512u * gl_WorkGroupID.x;
    uint _47 = sort_params._m0[5u];
    for (uint _49 = 0u; _49 < 4u; )
    {
        uint _55 = _49 * 128u;
        uint _57 = (_45 + _55) + gl_LocalInvocationID.x;
        uint _59 = _55 + gl_LocalInvocationID.x;
        gs_FFX_PARALLELSORT_LDS[_59 % 4u][_59 / 4u] = (_57 < _47) ? rw_scan_source._m0[_57] : 0u;
        _49++;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _70;
    _70 = 0u;
    for (uint _73 = 0u; _73 < 4u; )
    {
        uint _79 = gs_FFX_PARALLELSORT_LDS[_73][gl_LocalInvocationID.x];
        gs_FFX_PARALLELSORT_LDS[_73][gl_LocalInvocationID.x] = _70;
        _70 += _79;
        _73++;
        continue;
    }
    uint _81 = simd_prefix_exclusive_sum(_70);
    uint _83 = gl_LocalInvocationID.x / gl_SubgroupSize;
    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
    {
        gs_FFX_PARALLELSORT_LDSSums[_83] = _81 + _70;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(_83 != 0u))
    {
        gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _102 = _81 + gs_FFX_PARALLELSORT_LDSSums[_83];
    for (uint _105 = 0u; _105 < 4u; )
    {
        gs_FFX_PARALLELSORT_LDS[_105][gl_LocalInvocationID.x] += _102;
        _105++;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint _115 = 0u; _115 < 4u; _115++)
    {
        uint _121 = _115 * 128u;
        uint _123 = (_45 + _121) + gl_LocalInvocationID.x;
        uint _125 = _121 + gl_LocalInvocationID.x;
        if (_123 < _47)
        {
            rw_scan_dest._m0[_123] = gs_FFX_PARALLELSORT_LDS[_125 % 4u][_125 / 4u];
        }
    }
}


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

kernel void FfxParallelSortScanAdd(const device type_StructuredBuffer_uint& sort_params [[buffer(0)]], device type_RWStructuredBuffer_uint& rw_scan_source [[buffer(1)]], device type_RWStructuredBuffer_uint& rw_scan_dest [[buffer(2)]], device type_RWStructuredBuffer_uint& rw_scan_scratch [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_LDSSums;
    threadgroup spvUnsafeArray<spvUnsafeArray<uint, 128>, 4> gs_FFX_PARALLELSORT_LDS;
    do
    {
        if (gl_WorkGroupID.x >= (sort_params._m0[4u] * 16u))
        {
            break;
        }
        uint _55 = sort_params._m0[4u];
        uint _58 = sort_params._m0[2u];
        uint _59 = (gl_WorkGroupID.x / _55) * _58;
        uint _61 = sort_params._m0[4u];
        uint _63 = (gl_WorkGroupID.x % _61) * 512u;
        uint _65 = sort_params._m0[2u];
        for (uint _67 = 0u; _67 < 4u; )
        {
            uint _73 = _67 * 128u;
            uint _75 = (_63 + _73) + gl_LocalInvocationID.x;
            uint _77 = _73 + gl_LocalInvocationID.x;
            gs_FFX_PARALLELSORT_LDS[_77 % 4u][_77 / 4u] = (_75 < _65) ? rw_scan_source._m0[_59 + _75] : 0u;
            _67++;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _89;
        _89 = 0u;
        for (uint _92 = 0u; _92 < 4u; )
        {
            uint _98 = gs_FFX_PARALLELSORT_LDS[_92][gl_LocalInvocationID.x];
            gs_FFX_PARALLELSORT_LDS[_92][gl_LocalInvocationID.x] = _89;
            _89 += _98;
            _92++;
            continue;
        }
        uint _100 = simd_prefix_exclusive_sum(_89);
        uint _102 = gl_LocalInvocationID.x / gl_SubgroupSize;
        if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
        {
            gs_FFX_PARALLELSORT_LDSSums[_102] = _100 + _89;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!(_102 != 0u))
        {
            gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _121 = _100 + gs_FFX_PARALLELSORT_LDSSums[_102];
        uint _124 = rw_scan_scratch._m0[gl_WorkGroupID.x];
        for (uint _127 = 0u; _127 < 4u; )
        {
            gs_FFX_PARALLELSORT_LDS[_127][gl_LocalInvocationID.x] += _121;
            _127++;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint _137 = 0u; _137 < 4u; _137++)
        {
            uint _143 = _137 * 128u;
            uint _145 = (_63 + _143) + gl_LocalInvocationID.x;
            uint _147 = _143 + gl_LocalInvocationID.x;
            if (_145 < _65)
            {
                rw_scan_dest._m0[_59 + _145] = gs_FFX_PARALLELSORT_LDS[_147 % 4u][_147 / 4u] + _124;
            }
        }
        break;
    } while(false);
}


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

kernel void FfxParallelSortReduce(const device type_StructuredBuffer_uint& sort_params [[buffer(0)]], device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(1)]], device type_RWStructuredBuffer_uint& rw_reduce_table [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]])
{
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_LDSSums;
    do
    {
        if (gl_WorkGroupID.x >= (sort_params._m0[4u] * 16u))
        {
            break;
        }
        uint _52 = (gl_WorkGroupID.x / sort_params._m0[4u]) * sort_params._m0[2u];
        uint _54 = (gl_WorkGroupID.x % sort_params._m0[4u]) * 512u;
        uint _56;
        _56 = 0u;
        for (uint _59 = 0u; _59 < 4u; )
        {
            uint _66 = (_54 + (_59 * 128u)) + gl_LocalInvocationID.x;
            _56 += ((_66 < sort_params._m0[2u]) ? rw_sum_table._m0[_52 + _66] : 0u);
            _59++;
            continue;
        }
        uint _72 = simd_sum(_56);
        uint _74 = gl_LocalInvocationID.x / gl_SubgroupSize;
        bool _75 = simd_is_first();
        if (_75)
        {
            gs_FFX_PARALLELSORT_LDSSums[_74] = _72;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _90;
        if (!(_74 != 0u))
        {
            _90 = simd_sum((gl_LocalInvocationID.x < (128u / gl_SubgroupSize)) ? gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x] : 0u);
        }
        else
        {
            _90 = _72;
        }
        if (gl_LocalInvocationID.x == 0u)
        {
            rw_reduce_table._m0[gl_WorkGroupID.x] = _90;
        }
        break;
    } while(false);
}


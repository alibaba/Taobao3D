#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

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
    float4 _Packed_Params1;
};

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void FfxParallelSortCount(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& sort_params [[buffer(1)]], device type_RWStructuredBuffer_uint& rw_source_keys [[buffer(2)]], device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<uint, 2048> gs_FFX_PARALLELSORT_Histogram;
    uint _332 = as_type<uint>(Constants._Packed_Params1.z);
    do
    {
        uint _500 = sort_params._m0[2u];
        if (gl_WorkGroupID.x >= _500)
        {
            break;
        }
        for (int _538 = 0; _538 < 16; )
        {
            gs_FFX_PARALLELSORT_Histogram[uint(_538 * 128) + gl_LocalInvocationID.x] = 0u;
            _538++;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _379 = uint(int(sort_params._m0[1u]));
        uint _387 = (512u * _379) * gl_WorkGroupID.x;
        uint _392 = _500 - sort_params._m0[3u];
        uint _541;
        uint _546;
        if (gl_WorkGroupID.x >= _392)
        {
            _546 = _379 + 1u;
            _541 = _387 + ((gl_WorkGroupID.x - _392) * 512u);
        }
        else
        {
            _546 = _379;
            _541 = _387;
        }
        uint _410 = _541 + gl_LocalInvocationID.x;
        spvUnsafeArray<uint, 4> _345;
        for (uint _544 = 0u, _557 = _410; _544 < _546; _557 += 512u, _544++)
        {
            _345[0] = rw_source_keys._m0[_557];
            _345[1] = rw_source_keys._m0[_557 + 128u];
            _345[2] = rw_source_keys._m0[_557 + 256u];
            _345[3] = rw_source_keys._m0[_557 + 384u];
            uint _583;
            for (uint _558 = 0u, _564 = _557; _558 < 4u; _564 = _583, _558++)
            {
                if (_564 < sort_params._m0[0u])
                {
                    uint _452 = atomic_fetch_add_explicit((threadgroup atomic_uint*)&gs_FFX_PARALLELSORT_Histogram[(((_345[_558] >> (_332 & 31u)) & 15u) * 128u) + gl_LocalInvocationID.x], 1u, memory_order_relaxed);
                    _583 = _564 + 128u;
                }
                else
                {
                    _583 = _564;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_LocalInvocationID.x < 16u)
        {
            uint _556;
            _556 = 0u;
            for (int _548 = 0; _548 < 128; )
            {
                _556 += gs_FFX_PARALLELSORT_Histogram[(gl_LocalInvocationID.x * 128u) + uint(_548)];
                _548++;
                continue;
            }
            rw_sum_table._m0[(gl_LocalInvocationID.x * _500) + gl_WorkGroupID.x] = _556;
        }
        break;
    } while(false);
}


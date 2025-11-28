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

kernel void FfxParallelSortScatter(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& sort_params [[buffer(1)]], device type_RWStructuredBuffer_uint& rw_source_keys [[buffer(2)]], device type_RWStructuredBuffer_uint& rw_dest_keys [[buffer(3)]], device type_RWStructuredBuffer_uint& rw_source_payloads [[buffer(4)]], device type_RWStructuredBuffer_uint& rw_dest_payloads [[buffer(5)]], device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(6)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_LDSSums;
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_BinOffsetCache;
    threadgroup spvUnsafeArray<uint, 16> gs_FFX_PARALLELSORT_LocalHistogram;
    threadgroup spvUnsafeArray<uint, 128> gs_FFX_PARALLELSORT_LDSScratch;
    uint _78 = as_type<uint>(Constants._Packed_Params1.z);
    do
    {
        uint _82 = sort_params._m0[2u];
        if (gl_WorkGroupID.x >= _82)
        {
            break;
        }
        uint _87 = sort_params._m0[1u];
        uint _89 = uint(int(_87));
        uint _91 = sort_params._m0[3u];
        uint _93 = sort_params._m0[0u];
        bool _94 = gl_LocalInvocationID.x < 16u;
        if (_94)
        {
            gs_FFX_PARALLELSORT_BinOffsetCache[gl_LocalInvocationID.x] = rw_sum_table._m0[(gl_LocalInvocationID.x * _82) + gl_WorkGroupID.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _103 = (512u * _89) * gl_WorkGroupID.x;
        uint _104 = _82 - _91;
        uint _113;
        uint _114;
        if (gl_WorkGroupID.x >= _104)
        {
            _113 = _89 + 1u;
            _114 = _103 + ((gl_WorkGroupID.x - _104) * 512u);
        }
        else
        {
            _113 = _89;
            _114 = _103;
        }
        uint _115 = _114 + gl_LocalInvocationID.x;
        spvUnsafeArray<uint, 4> _69;
        spvUnsafeArray<uint, 4> _70;
        for (uint _117 = _115, _120 = 0u; _120 < _113; _117 += 512u, _120++)
        {
            _69[0] = rw_source_keys._m0[_117];
            uint _128 = _117 + 128u;
            _69[1] = rw_source_keys._m0[_128];
            uint _132 = _117 + 256u;
            _69[2] = rw_source_keys._m0[_132];
            uint _136 = _117 + 384u;
            _69[3] = rw_source_keys._m0[_136];
            _70[0] = rw_source_payloads._m0[_117];
            _70[1] = rw_source_payloads._m0[_128];
            _70[2] = rw_source_payloads._m0[_132];
            _70[3] = rw_source_payloads._m0[_136];
            uint _157;
            uint _156 = _117;
            int _159 = 0;
            for (; _159 < 4; _156 = _157, _159++)
            {
                if (_94)
                {
                    gs_FFX_PARALLELSORT_LocalHistogram[gl_LocalInvocationID.x] = 0u;
                }
                bool _168 = _156 < _93;
                uint _177;
                uint _180;
                _177 = _168 ? _70[_159] : 0u;
                _180 = _168 ? _69[_159] : 4294967295u;
                uint _178;
                uint _181;
                for (uint _182 = 0u; _182 < 4u; _177 = _178, _180 = _181, _182 += 2u)
                {
                    uint _188 = _180 >> (_78 & 31u);
                    uint _190 = _182 & 31u;
                    uint _194 = ((((_188 & 15u) >> _190) & 3u) * 8u) & 31u;
                    uint _195 = 1u << _194;
                    uint _196 = simd_prefix_exclusive_sum(_195);
                    uint _198 = gl_LocalInvocationID.x / gl_SubgroupSize;
                    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
                    {
                        gs_FFX_PARALLELSORT_LDSSums[_198] = _196 + _195;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (!(_198 != 0u))
                    {
                        gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x] = simd_prefix_exclusive_sum(gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint _216 = gs_FFX_PARALLELSORT_LDSSums[_198];
                    uint _217 = _196 + _216;
                    if (gl_LocalInvocationID.x == 127u)
                    {
                        gs_FFX_PARALLELSORT_LDSScratch[0] = _217 + _195;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint _224 = gs_FFX_PARALLELSORT_LDSScratch[0];
                    uint _234 = ((_217 + (((_224 << 8u) + (_224 << 16u)) + (_224 << 24u))) >> _194) & 255u;
                    gs_FFX_PARALLELSORT_LDSSums[_234] = _180;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    _181 = gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    gs_FFX_PARALLELSORT_LDSSums[_234] = _177;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    _178 = gs_FFX_PARALLELSORT_LDSSums[gl_LocalInvocationID.x];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                uint _240 = _180 >> (_78 & 31u);
                uint _241 = _240 & 15u;
                uint _243 = atomic_fetch_add_explicit((threadgroup atomic_uint*)&gs_FFX_PARALLELSORT_LocalHistogram[_241], 1u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint _248 = simd_prefix_exclusive_sum(_94 ? gs_FFX_PARALLELSORT_LocalHistogram[gl_LocalInvocationID.x] : 0u);
                if (_94)
                {
                    gs_FFX_PARALLELSORT_LDSScratch[gl_LocalInvocationID.x] = _248;
                }
                uint _254 = gs_FFX_PARALLELSORT_BinOffsetCache[_241];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint _258 = _254 + (gl_LocalInvocationID.x - gs_FFX_PARALLELSORT_LDSScratch[_241]);
                if (_258 < _93)
                {
                    rw_dest_keys._m0[_258] = _180;
                    rw_dest_payloads._m0[_258] = _177;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (_94)
                {
                    gs_FFX_PARALLELSORT_BinOffsetCache[gl_LocalInvocationID.x] += gs_FFX_PARALLELSORT_LocalHistogram[gl_LocalInvocationID.x];
                }
                _157 = _156 + 128u;
            }
        }
        break;
    } while(false);
}


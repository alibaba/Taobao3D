#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _PointCount;
};

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

kernel void GenIndirectDrawArgs(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _VisibleMaskPrefixSums [[buffer(1)]], device type_RWStructuredBuffer_uint& _IndirectDrawArgs [[buffer(2)]], device type_RWStructuredBuffer_uint& _SortParams [[buffer(3)]], device type_RWStructuredBuffer_uint& _SortSumPassIndirectBuffer [[buffer(4)]], device type_RWStructuredBuffer_uint& _SortReducePassIndirectBuffer [[buffer(5)]], device type_RWStructuredBuffer_uint& _SortCopyPassIndirectBuffer [[buffer(6)]])
{
    uint _157 = as_type<uint>(Constants._PointCount.x) - 1u;
    uint _159 = _VisibleMaskPrefixSums._m0[_157];
    _IndirectDrawArgs._m0[0u] = 6u;
    _IndirectDrawArgs._m0[1u] = _159 * 2u;
    _IndirectDrawArgs._m0[2u] = 0u;
    _IndirectDrawArgs._m0[3u] = 0u;
    _IndirectDrawArgs._m0[4u] = 0u;
    _IndirectDrawArgs._m0[5u] = 0u;
    _IndirectDrawArgs._m0[6u] = 0u;
    _IndirectDrawArgs._m0[7u] = 0u;
    uint _191 = (_159 + 511u) / 512u;
    bool _200 = _191 < 800u;
    uint _249 = _200 ? _191 : 800u;
    uint _209 = (_249 + 511u) / 512u;
    _SortParams._m0[0u] = _159;
    _SortParams._m0[1u] = _200 ? 1u : (_191 / 800u);
    _SortParams._m0[2u] = _249;
    _SortParams._m0[3u] = _200 ? 0u : (_191 % 800u);
    _SortParams._m0[4u] = _209;
    _SortParams._m0[5u] = 16u * _209;
    _SortParams._m0[6u] = 0u;
    _SortParams._m0[7u] = 0u;
    _SortSumPassIndirectBuffer._m0[0u] = _249;
    _SortSumPassIndirectBuffer._m0[1u] = 1u;
    _SortSumPassIndirectBuffer._m0[2u] = 1u;
    _SortReducePassIndirectBuffer._m0[0u] = _209 * 16u;
    _SortReducePassIndirectBuffer._m0[1u] = 1u;
    _SortReducePassIndirectBuffer._m0[2u] = 1u;
    _SortCopyPassIndirectBuffer._m0[0u] = (_159 + 255u) / 256u;
    _SortCopyPassIndirectBuffer._m0[1u] = 1u;
    _SortCopyPassIndirectBuffer._m0[2u] = 1u;
}


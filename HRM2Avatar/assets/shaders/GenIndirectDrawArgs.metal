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

kernel void GenIndirectDrawArgs(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_uint& _VisibleMaskPrefixSums [[buffer(1)]], device type_RWStructuredBuffer_uint& _IndirectDrawArgs [[buffer(2)]], device type_RWStructuredBuffer_uint& _SortParams [[buffer(3)]])
{
    uint _133 = as_type<uint>(Constants._PointCount.x) - 1u;
    uint _135 = _VisibleMaskPrefixSums._m0[_133];
    _IndirectDrawArgs._m0[0u] = 6u;
    _IndirectDrawArgs._m0[1u] = _135;
    _IndirectDrawArgs._m0[2u] = 0u;
    _IndirectDrawArgs._m0[3u] = 0u;
    _IndirectDrawArgs._m0[4u] = 0u;
    _IndirectDrawArgs._m0[5u] = 0u;
    _IndirectDrawArgs._m0[6u] = 0u;
    _IndirectDrawArgs._m0[7u] = 0u;
    uint _166 = (_135 + 511u) / 512u;
    bool _175 = _166 < 800u;
    uint _209 = _175 ? _166 : 800u;
    uint _184 = (_209 + 511u) / 512u;
    _SortParams._m0[0u] = _135;
    _SortParams._m0[1u] = _175 ? 1u : (_166 / 800u);
    _SortParams._m0[2u] = _209;
    _SortParams._m0[3u] = _175 ? 0u : (_166 % 800u);
    _SortParams._m0[4u] = _184;
    _SortParams._m0[5u] = 16u * _184;
    _SortParams._m0[6u] = 0u;
    _SortParams._m0[7u] = 0u;
}


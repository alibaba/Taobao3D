#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _CustomScreenParams;
};

struct SplatProjData
{
    float4 pos;
    half4 color;
    half2 axis1;
    half2 axis2;
};

struct type_StructuredBuffer_SplatProjData
{
    SplatProjData _m0[1];
};

struct type_StructuredBuffer_uint
{
    uint _m0[1];
};

struct Vert_out
{
    half4 out_var_COLOR0 [[user(locn0)]];
    half2 out_var_TEXCOORD0 [[user(locn1)]];
    float4 gl_Position [[position]];
};

vertex Vert_out Vert(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(1)]], const device type_StructuredBuffer_uint& _VisiblePointList [[buffer(2)]], uint gl_VertexIndex [[vertex_id]], uint gl_InstanceIndex [[instance_id]])
{
    Vert_out out = {};
    float2 _243 = ((float2(float(gl_VertexIndex & 1u), float((gl_VertexIndex >> 1u) & 1u)) * 2.0) - float2(1.0)) * float(sqrt(log(half(0.0039215087890625) / _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].color.w) * half(-0.5)));
    float2 _274 = _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.xy + (((((float2(_SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].axis1) * _243.x) + (float2(_SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].axis2) * _243.y)) * 2.0) / Constants._CustomScreenParams.xy) * _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.w);
    float4 _277 = float4(_274.x, _274.y, _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.z, _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.w);
    _277.z = (_SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.z + _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].pos.w) * 0.5;
    _277.y = _274.y * (-1.0);
    out.gl_Position = _277;
    out.out_var_COLOR0 = _SplatProjData0._m0[_VisiblePointList._m0[gl_InstanceIndex]].color;
    out.out_var_TEXCOORD0 = half2(_243);
    return out;
}


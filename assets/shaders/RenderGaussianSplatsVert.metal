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

vertex Vert_out Vert(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(1)]], const device type_StructuredBuffer_uint& _OrderBuffer [[buffer(2)]], uint gl_VertexIndex [[vertex_id]], uint gl_InstanceIndex [[instance_id]])
{
    Vert_out out = {};
    half2 _385;
    half4 _386;
    float4 _387;
    do
    {
        if (_SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.w <= 0.0)
        {
            _387 = float4(as_type<float>(0x7fc00000u /* nan */));
            _386 = half4(half(0.0));
            _385 = half2(half(0.0));
            break;
        }
        float2 _284 = ((float2(float(gl_VertexIndex & 1u), float((gl_VertexIndex >> 1u) & 1u)) * 2.0) - float2(1.0)) * float(sqrt(log(half(0.0039215087890625) / _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].color.w) * half(-0.5)));
        float2 _315 = _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.xy + (((((float2(_SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].axis1) * _284.x) + (float2(_SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].axis2) * _284.y)) * 2.0) / Constants._CustomScreenParams.xy) * _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.w);
        float4 _318 = float4(_315.x, _315.y, _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.z, _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.w);
        _318.z = (_SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.z + _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].pos.w) * 0.5;
        _318.y = _315.y * (-1.0);
        _387 = _318;
        _386 = _SplatProjData0._m0[_OrderBuffer._m0[gl_InstanceIndex]].color;
        _385 = half2(_284);
        break;
    } while(false);
    out.gl_Position = _387;
    out.out_var_COLOR0 = _386;
    out.out_var_TEXCOORD0 = _385;
    return out;
}


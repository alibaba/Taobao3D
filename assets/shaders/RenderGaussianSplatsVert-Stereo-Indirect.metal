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
    uint gl_Layer [[render_target_array_index]];
};

vertex Vert_out Vert(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(1)]], const device type_StructuredBuffer_SplatProjData& _SplatProjData1 [[buffer(2)]], const device type_StructuredBuffer_uint& _VisiblePointList [[buffer(3)]], uint gl_VertexIndex [[vertex_id]], uint gl_InstanceIndex [[instance_id]])
{
    Vert_out out = {};
    uint _261;
    half2 _453;
    half4 _454;
    float4 _455;
    do
    {
        _261 = gl_InstanceIndex & 1u;
        uint _266 = gl_InstanceIndex >> 1u;
        float4 _445;
        half4 _446;
        half2 _448;
        half2 _449;
        if (_261 == 0u)
        {
            _449 = _SplatProjData0._m0[_VisiblePointList._m0[_266]].axis2;
            _448 = _SplatProjData0._m0[_VisiblePointList._m0[_266]].axis1;
            _446 = _SplatProjData0._m0[_VisiblePointList._m0[_266]].color;
            _445 = _SplatProjData0._m0[_VisiblePointList._m0[_266]].pos;
        }
        else
        {
            _449 = _SplatProjData1._m0[_VisiblePointList._m0[_266]].axis2;
            _448 = _SplatProjData1._m0[_VisiblePointList._m0[_266]].axis1;
            _446 = _SplatProjData1._m0[_VisiblePointList._m0[_266]].color;
            _445 = _SplatProjData1._m0[_VisiblePointList._m0[_266]].pos;
        }
        if (_445.w <= 0.0)
        {
            _455 = float4(as_type<float>(0x7fc00000u /* nan */));
            _454 = half4(half(0.0));
            _453 = half2(half(0.0));
            break;
        }
        float2 _329 = ((float2(float(gl_VertexIndex & 1u), float((gl_VertexIndex >> 1u) & 1u)) * 2.0) - float2(1.0)) * float(sqrt(log(half(0.0039215087890625) / _446.w) * half(-0.5)));
        float2 _360 = _445.xy + (((((float2(_448) * _329.x) + (float2(_449) * _329.y)) * 2.0) / Constants._CustomScreenParams.xy) * _445.w);
        float4 _363 = float4(_360.x, _360.y, _445.z, _445.w);
        _363.z = (_445.z + _445.w) * 0.5;
        _363.y = _360.y * (-1.0);
        _455 = _363;
        _454 = _446;
        _453 = half2(_329);
        break;
    } while(false);
    out.gl_Position = _455;
    out.out_var_COLOR0 = _454;
    out.out_var_TEXCOORD0 = _453;
    out.gl_Layer = _261;
    return out;
}


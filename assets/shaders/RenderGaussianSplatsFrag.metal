#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Frag_out
{
    half4 out_var_SV_Target [[color(0)]];
};

struct Frag_in
{
    half4 in_var_COLOR0 [[user(locn0)]];
    half2 in_var_TEXCOORD0 [[user(locn1)]];
};

fragment Frag_out Frag(Frag_in in [[stage_in]])
{
    Frag_out out = {};
    half _121 = clamp(exp(-dot(in.in_var_TEXCOORD0, in.in_var_TEXCOORD0)) * in.in_var_COLOR0.w, half(0.0), half(1.0));
    out.out_var_SV_Target = half4(in.in_var_COLOR0.xyz * _121, _121);
    return out;
}


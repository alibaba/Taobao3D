#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Vert_out
{
    float2 out_var_TEXCOORD0 [[user(locn0)]];
    float4 gl_Position [[position]];
};

vertex Vert_out Vert(uint gl_VertexIndex [[vertex_id]])
{
    Vert_out out = {};
    float _89 = float(gl_VertexIndex & 1u);
    uint _92 = gl_VertexIndex >> 1u;
    out.gl_Position = float4((float2(_89, float(_92 & 1u)) * 2.0) - float2(1.0), 0.5, 1.0);
    out.out_var_TEXCOORD0 = float2(_89, float(_92 ^ 1u));
    return out;
}


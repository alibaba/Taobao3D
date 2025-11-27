#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Vert_out
{
    float2 out_var_TEXCOORD0 [[user(locn0)]];
    float4 gl_Position [[position]];
    uint gl_Layer [[render_target_array_index]];
};

vertex Vert_out Vert(uint gl_VertexIndex [[vertex_id]], uint gl_InstanceIndex [[instance_id]])
{
    Vert_out out = {};
    float _109 = float(gl_VertexIndex & 1u);
    uint _112 = gl_VertexIndex >> 1u;
    out.gl_Position = float4((float2(_109, float(_112 & 1u)) * 2.0) - float2(1.0), 0.5, 1.0);
    out.out_var_TEXCOORD0 = float2(_109, float(_112 ^ 1u));
    out.gl_Layer = gl_InstanceIndex & 1u;
    return out;
}


#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4x4 _MVP[2];
};

struct Vert_out
{
    float4 gl_Position [[position]];
    uint gl_Layer [[render_target_array_index]];
};

struct Vert_in
{
    float3 in_var_POSITION0 [[attribute(0)]];
};

vertex Vert_out Vert(Vert_in in [[stage_in]], constant type_Constants& Constants [[buffer(0)]], uint gl_InstanceIndex [[instance_id]])
{
    Vert_out out = {};
    uint _75 = gl_InstanceIndex & 1u;
    out.gl_Position = Constants._MVP[_75] * float4(in.in_var_POSITION0, 1.0);
    out.gl_Layer = _75;
    return out;
}


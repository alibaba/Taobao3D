#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Frag_out
{
    half4 out_var_SV_Target [[color(0)]];
};

struct Frag_in
{
    float2 in_var_TEXCOORD0 [[user(locn0)]];
};

fragment Frag_out Frag(Frag_in in [[stage_in]], texture2d_array<half> _InTexture [[texture(0)]], sampler _Sampler [[sampler(0)]], uint gl_Layer [[render_target_array_index]])
{
    Frag_out out = {};
    float3 _127 = float3(in.in_var_TEXCOORD0, float(gl_Layer));
    half4 _129 = _InTexture.sample(_Sampler, _127.xy, uint(rint(_127.z)));
    float3 _132 = float3(_129.xyz);
    float _144 = _132.x;
    float _147 = _132.y;
    float _150 = _132.z;
    half3 _134 = half3(float3((_144 <= 0.040449999272823333740234375) ? (_144 * 0.077399380505084991455078125) : powr((_144 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_147 <= 0.040449999272823333740234375) ? (_147 * 0.077399380505084991455078125) : powr((_147 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_150 <= 0.040449999272823333740234375) ? (_150 * 0.077399380505084991455078125) : powr((_150 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625)));
    out.out_var_SV_Target = half4(_134.x, _134.y, _134.z, _129.w);
    return out;
}


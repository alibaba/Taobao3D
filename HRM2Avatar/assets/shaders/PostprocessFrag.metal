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

fragment Frag_out Frag(Frag_in in [[stage_in]], texture2d<half> _InTexture [[texture(0)]], sampler _Sampler [[sampler(0)]])
{
    Frag_out out = {};
    half4 _112 = _InTexture.sample(_Sampler, in.in_var_TEXCOORD0);
    float3 _115 = float3(_112.xyz);
    float _127 = _115.x;
    float _130 = _115.y;
    float _133 = _115.z;
    half3 _117 = half3(float3((_127 <= 0.040449999272823333740234375) ? (_127 * 0.077399380505084991455078125) : powr((_127 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_130 <= 0.040449999272823333740234375) ? (_130 * 0.077399380505084991455078125) : powr((_130 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_133 <= 0.040449999272823333740234375) ? (_133 * 0.077399380505084991455078125) : powr((_133 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625)));
    out.out_var_SV_Target = half4(_117.x, _117.y, _117.z, _112.w);
    return out;
}


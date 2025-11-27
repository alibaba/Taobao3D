#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _Params;
};

kernel void Postprocess(constant type_Constants& Constants [[buffer(0)]], texture2d<float> _InTexture [[texture(0)]], texture2d<float, access::write> _OutTexture [[texture(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    switch (0u)
    {
        default:
        {
            if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._Params.z))
            {
                break;
            }
            uint _161 = as_type<uint>(Constants._Params.x);
            int _171 = int(gl_GlobalInvocationID.x % _161);
            int _173 = int(gl_GlobalInvocationID.x / _161);
            float4 _178 = _InTexture.read(uint2(int3(_171, _173, 0).xy), 0);
            float _197 = _178.x;
            float _200 = _178.y;
            float _203 = _178.z;
            float3 _205 = float3((_197 <= 0.040449999272823333740234375) ? (_197 * 0.077399380505084991455078125) : pow((_197 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_200 <= 0.040449999272823333740234375) ? (_200 * 0.077399380505084991455078125) : pow((_200 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_203 <= 0.040449999272823333740234375) ? (_203 * 0.077399380505084991455078125) : pow((_203 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625));
            _OutTexture.write(float4(_205.x, _205.y, _205.z, _178.w), uint2(uint2(int2(_171, _173))));
            break;
        }
    }
}


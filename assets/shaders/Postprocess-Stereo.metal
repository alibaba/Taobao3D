#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _Params;
};

kernel void Postprocess(constant type_Constants& Constants [[buffer(0)]], texture2d_array<float> _InTexture [[texture(0)]], texture2d_array<float, access::write> _OutTexture [[texture(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    switch (0u)
    {
        default:
        {
            if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._Params.z))
            {
                break;
            }
            uint _166 = as_type<uint>(Constants._Params.x);
            int _176 = int(gl_GlobalInvocationID.x % _166);
            int _178 = int(gl_GlobalInvocationID.x / _166);
            int _181 = int(gl_GlobalInvocationID.z);
            int3 _183 = int4(_176, _178, _181, 0).xyz;
            float4 _186 = _InTexture.read(uint2(_183.xy), uint(_183.z), 0);
            float _208 = _186.x;
            float _211 = _186.y;
            float _214 = _186.z;
            float3 _216 = float3((_208 <= 0.040449999272823333740234375) ? (_208 * 0.077399380505084991455078125) : pow((_208 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_211 <= 0.040449999272823333740234375) ? (_211 * 0.077399380505084991455078125) : pow((_211 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625), (_214 <= 0.040449999272823333740234375) ? (_214 * 0.077399380505084991455078125) : pow((_214 + 0.054999999701976776123046875) * 0.947867333889007568359375, 2.400000095367431640625));
            uint3 _201 = uint3(int3(_176, _178, _181));
            _OutTexture.write(float4(_216.x, _216.y, _216.z, _186.w), uint2(_201.xy), uint(_201.z));
            break;
        }
    }
}


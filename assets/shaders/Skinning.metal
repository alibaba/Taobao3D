#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4 _MeshLayout;
    float4 _BonePositions[200];
    float4 _BoneRotations[200];
};

struct type_ByteAddressBuffer
{
    uint _m0[1];
};

struct type_RWByteAddressBuffer
{
    uint _m0[1];
};

kernel void Skinning(constant type_Constants& Constants [[buffer(0)]], const device type_ByteAddressBuffer& _MeshVertice [[buffer(1)]], device type_RWByteAddressBuffer& _MeshVerticeDst [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._MeshLayout.w))
        {
            break;
        }
        int _466 = as_type<int>(Constants._MeshLayout.x) * int(gl_GlobalInvocationID.x);
        uint _469 = uint(_466) >> 2u;
        uint _471 = _MeshVertice._m0[_469];
        uint _472 = _469 + 1u;
        uint _474 = _MeshVertice._m0[_472];
        uint _475 = _469 + 2u;
        uint _477 = _MeshVertice._m0[_475];
        int _495 = int(_MeshVertice._m0[uint(_466 + as_type<int>(Constants._MeshLayout.z)) >> 2u]);
        int _545 = (_495 >> 0) & 255;
        int _550 = (_495 >> 8) & 255;
        int _555 = (_495 >> 16) & 255;
        int _560 = (_495 >> 24) & 255;
        uint _505 = uint(_466 + as_type<int>(Constants._MeshLayout.y)) >> 2u;
        float4 _518 = as_type<float4>(uint4(_MeshVertice._m0[_505], _MeshVertice._m0[_505 + 1u], _MeshVertice._m0[_505 + 2u], _MeshVertice._m0[_505 + 3u]));
        float _669 = Constants._BoneRotations[_545].x * 2.0;
        float _672 = Constants._BoneRotations[_545].y * 2.0;
        float _675 = Constants._BoneRotations[_545].z * 2.0;
        float _687 = Constants._BoneRotations[_545].z * _675;
        float _703 = Constants._BoneRotations[_545].w * _669;
        float _707 = Constants._BoneRotations[_545].w * _672;
        float _711 = Constants._BoneRotations[_545].w * _675;
        float _820 = Constants._BoneRotations[_550].x * 2.0;
        float _823 = Constants._BoneRotations[_550].y * 2.0;
        float _826 = Constants._BoneRotations[_550].z * 2.0;
        float _838 = Constants._BoneRotations[_550].z * _826;
        float _854 = Constants._BoneRotations[_550].w * _820;
        float _858 = Constants._BoneRotations[_550].w * _823;
        float _862 = Constants._BoneRotations[_550].w * _826;
        float _971 = Constants._BoneRotations[_555].x * 2.0;
        float _974 = Constants._BoneRotations[_555].y * 2.0;
        float _977 = Constants._BoneRotations[_555].z * 2.0;
        float _989 = Constants._BoneRotations[_555].z * _977;
        float _1005 = Constants._BoneRotations[_555].w * _971;
        float _1009 = Constants._BoneRotations[_555].w * _974;
        float _1013 = Constants._BoneRotations[_555].w * _977;
        float _1122 = Constants._BoneRotations[_560].x * 2.0;
        float _1125 = Constants._BoneRotations[_560].y * 2.0;
        float _1128 = Constants._BoneRotations[_560].z * 2.0;
        float _1140 = Constants._BoneRotations[_560].z * _1128;
        float _1156 = Constants._BoneRotations[_560].w * _1122;
        float _1160 = Constants._BoneRotations[_560].w * _1125;
        float _1164 = Constants._BoneRotations[_560].w * _1128;
        float4x4 _588 = float4x4(float4(1.0 - fma(Constants._BoneRotations[_545].y, _672, _687), fma(Constants._BoneRotations[_545].x, _672, -_711), fma(Constants._BoneRotations[_545].x, _675, _707), Constants._BonePositions[_545].x), float4(fma(Constants._BoneRotations[_545].x, _672, _711), 1.0 - fma(Constants._BoneRotations[_545].x, _669, _687), fma(Constants._BoneRotations[_545].y, _675, -_703), Constants._BonePositions[_545].y), float4(fma(Constants._BoneRotations[_545].x, _675, -_707), fma(Constants._BoneRotations[_545].y, _675, _703), 1.0 - fma(Constants._BoneRotations[_545].x, _669, Constants._BoneRotations[_545].y * _672), Constants._BonePositions[_545].z), float4(0.0, 0.0, 0.0, Constants._BonePositions[_545].w)) * _518.x;
        float4x4 _592 = float4x4(float4(1.0 - fma(Constants._BoneRotations[_550].y, _823, _838), fma(Constants._BoneRotations[_550].x, _823, -_862), fma(Constants._BoneRotations[_550].x, _826, _858), Constants._BonePositions[_550].x), float4(fma(Constants._BoneRotations[_550].x, _823, _862), 1.0 - fma(Constants._BoneRotations[_550].x, _820, _838), fma(Constants._BoneRotations[_550].y, _826, -_854), Constants._BonePositions[_550].y), float4(fma(Constants._BoneRotations[_550].x, _826, -_858), fma(Constants._BoneRotations[_550].y, _826, _854), 1.0 - fma(Constants._BoneRotations[_550].x, _820, Constants._BoneRotations[_550].y * _823), Constants._BonePositions[_550].z), float4(0.0, 0.0, 0.0, Constants._BonePositions[_550].w)) * _518.y;
        float4x4 _596 = float4x4(float4(1.0 - fma(Constants._BoneRotations[_555].y, _974, _989), fma(Constants._BoneRotations[_555].x, _974, -_1013), fma(Constants._BoneRotations[_555].x, _977, _1009), Constants._BonePositions[_555].x), float4(fma(Constants._BoneRotations[_555].x, _974, _1013), 1.0 - fma(Constants._BoneRotations[_555].x, _971, _989), fma(Constants._BoneRotations[_555].y, _977, -_1005), Constants._BonePositions[_555].y), float4(fma(Constants._BoneRotations[_555].x, _977, -_1009), fma(Constants._BoneRotations[_555].y, _977, _1005), 1.0 - fma(Constants._BoneRotations[_555].x, _971, Constants._BoneRotations[_555].y * _974), Constants._BonePositions[_555].z), float4(0.0, 0.0, 0.0, Constants._BonePositions[_555].w)) * _518.z;
        float4x4 _600 = float4x4(float4(1.0 - fma(Constants._BoneRotations[_560].y, _1125, _1140), fma(Constants._BoneRotations[_560].x, _1125, -_1164), fma(Constants._BoneRotations[_560].x, _1128, _1160), Constants._BonePositions[_560].x), float4(fma(Constants._BoneRotations[_560].x, _1125, _1164), 1.0 - fma(Constants._BoneRotations[_560].x, _1122, _1140), fma(Constants._BoneRotations[_560].y, _1128, -_1156), Constants._BonePositions[_560].y), float4(fma(Constants._BoneRotations[_560].x, _1128, -_1160), fma(Constants._BoneRotations[_560].y, _1128, _1156), 1.0 - fma(Constants._BoneRotations[_560].x, _1122, Constants._BoneRotations[_560].y * _1125), Constants._BonePositions[_560].z), float4(0.0, 0.0, 0.0, Constants._BonePositions[_560].w)) * _518.w;
        uint3 _529 = as_type<uint3>((float4(as_type<float3>(uint3(_471, _474, _477)), 1.0) * float4x4(((_588[0] + _592[0]) + _596[0]) + _600[0], ((_588[1] + _592[1]) + _596[1]) + _600[1], ((_588[2] + _592[2]) + _596[2]) + _600[2], ((_588[3] + _592[3]) + _596[3]) + _600[3])).xyz);
        _MeshVerticeDst._m0[_469] = _529.x;
        _MeshVerticeDst._m0[_472] = _529.y;
        _MeshVerticeDst._m0[_475] = _529.z;
        break;
    } while(false);
}


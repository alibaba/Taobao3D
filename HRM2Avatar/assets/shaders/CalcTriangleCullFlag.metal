#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4x4 _MatrixWorldToObject;
    float4 _CameraPosWS[2];
    float4 _MeshLayout;
};

struct type_StructuredBuffer_int
{
    int _m0[1];
};

struct type_ByteAddressBuffer
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_int
{
    int _m0[1];
};

kernel void CalcTriangleCullFlag(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_int& _MeshIndice [[buffer(1)]], const device type_ByteAddressBuffer& _MeshVertice [[buffer(2)]], const device type_ByteAddressBuffer& _GSTriangleProp [[buffer(3)]], device type_RWStructuredBuffer_int& _TriangleCullFlag [[buffer(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._MeshLayout.y))
        {
            break;
        }
        uint _65 = 3u * gl_GlobalInvocationID.x;
        int _79 = as_type<int>(Constants._MeshLayout.x);
        uint _82 = uint(_79 * _MeshIndice._m0[_65 + 2u]) >> 2u;
        float3 _92 = as_type<float3>(uint3(_MeshVertice._m0[_82], _MeshVertice._m0[_82 + 1u], _MeshVertice._m0[_82 + 2u]));
        uint _95 = uint(_79 * _MeshIndice._m0[_65 + 1u]) >> 2u;
        float3 _105 = as_type<float3>(uint3(_MeshVertice._m0[_95], _MeshVertice._m0[_95 + 1u], _MeshVertice._m0[_95 + 2u]));
        uint _108 = uint(_79 * _MeshIndice._m0[_65]) >> 2u;
        uint _131 = ((_GSTriangleProp._m0[(gl_GlobalInvocationID.x & 4294967292u) >> 2u] >> ((8u * (gl_GlobalInvocationID.x % 4u)) & 31u)) >> 1u) & 3u;
        _TriangleCullFlag._m0[gl_GlobalInvocationID.x] = int((_131 != 3u) && ((_131 == 0u) || ((int(dot(_92 - (Constants._MatrixWorldToObject * Constants._CameraPosWS[0]).xyz, cross(_105 - _92, as_type<float3>(uint3(_MeshVertice._m0[_108], _MeshVertice._m0[_108 + 1u], _MeshVertice._m0[_108 + 2u])) - _105)) >= 0.0) ^ int(_131 == 2u)) != 0))) - 1;
        break;
    } while(false);
}


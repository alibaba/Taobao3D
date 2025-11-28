#include "common.hlsl"

cbuffer Constants : register(b0)
{
    float4x4 _MatrixWorldToObject;
    float4 _CameraPosWS[2];
    float4 _MeshLayout;      // x: vertex stride, y: triangle count
};

StructuredBuffer<int> _MeshIndice : register(t1);
ByteAddressBuffer _MeshVertice : register(t2);
ByteAddressBuffer _GSTriangleProp : register(t3);

RWStructuredBuffer<int> _TriangleCullFlag : register(u4);

uint UnpackTriangleFrontFaceFromByte(ByteAddressBuffer buffer, uint byteIndex)
{
    uint packedData = buffer.Load(byteIndex & 0xFFFFFFFC);
    uint byteOffset = byteIndex % 4;
    uint targetByte = packedData >> (8 * byteOffset);
    uint value = (targetByte >> 1) & 0x03;
    return value;
}

[numthreads(GROUP_SIZE,1,1)]
void CalcTriangleCullFlag(uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    if (idx >= asuint(_MeshLayout.y))
        return;

    int mid0 = _MeshIndice[3 * idx];
    int mid1 = _MeshIndice[3 * idx + 1];
    int mid2 = _MeshIndice[3 * idx + 2];

    //align restrictly with training code
	int stride = asint(_MeshLayout.x);

	float3 v0 = asfloat(_MeshVertice.Load3(stride * mid2));
    float3 v1 = asfloat(_MeshVertice.Load3(stride * mid1));
    float3 v2 = asfloat(_MeshVertice.Load3(stride * mid0));

	float3 e1 = v1 - v0;
    float3 e2 = v2 - v1;
    float3 cross_res = cross(e1, e2);

    uint faceCull = UnpackTriangleFrontFaceFromByte(_GSTriangleProp, idx);
    float3 viewOS = v0 - mul(_MatrixWorldToObject, _CameraPosWS[0]).xyz;
    float ort = dot(viewOS, cross_res);

    // 3 always cull，2 cull front，1 cull back，0 always not cull
    bool cullFlag = faceCull != 3 && (faceCull == 0 || ((ort >= 0) ^ (faceCull == 2)));
#ifdef IS_STEREO_PIPELINE
    float3 viewOS1 = v0 - mul(_MatrixWorldToObject, _CameraPosWS[1]).xyz;
    float ort1 = dot(viewOS1, cross_res);
    cullFlag |= ((ort1 >= 0) ^ (faceCull == 2));
#endif // IS_STEREO_PIPELINE

    // -1: back, 0: front
    _TriangleCullFlag[idx] = cullFlag - 1;
}
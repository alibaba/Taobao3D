#include "common.hlsl"

#define MAX_BONES 200

cbuffer Constants : register(b0)
{
    float4 _MeshLayout;   //x vertex stride, y offset to skinning weight, z offset to skinning indices, w vertexCount

    float4 _BonePositions[MAX_BONES];  // cbuffer must align with 16 bytes
    float4 _BoneRotations[MAX_BONES];
};

ByteAddressBuffer _MeshVertice : register(t1);
RWByteAddressBuffer _MeshVerticeDst : register(u2);

int4 UnpackInt4FromInt32(int packedInt)
{
    int4 unpacked;
    unpacked[0] = (packedInt >> 0) & 0xFF; 
    unpacked[1] = (packedInt >> 8) & 0xFF;
    unpacked[2] = (packedInt >> 16) & 0xFF; 
    unpacked[3] = (packedInt >> 24) & 0xFF;
    return unpacked;
};

float4x4 CalcBoneMatrix(int index)
{
    float4 q = _BoneRotations[index];

    float x = q.x * 2.0;
    float y = q.y * 2.0;
    float z = q.z * 2.0;
    float xx = q.x * x;
    float yy = q.y * y;
    float zz = q.z * z;
    float xy = q.x * y;
    float xz = q.x * z;
    float yz = q.y * z;
    float wx = q.w * x;
    float wy = q.w * y;
    float wz = q.w * z;

    float4 c0;
    float4 c1;
    float4 c2;
    // Calculate 3x3 matrix from orthonormal basis
    c0.x = 1.0 - (yy + zz);
    c0.y = xy + wz;
    c0.z = xz - wy;
    c0.w = 0.0;

    c1.x = xy - wz;
    c1.y = 1.0 - (xx + zz);
    c1.z = yz + wx;
    c1.w = 0.0;

    c2.x = xz + wy;
    c2.y = yz - wx;
    c2.z = 1.0 - (xx + yy);
    c2.w = 0.0;

    float4 c3 = _BonePositions[index];

    return float4x4(c0.x, c1.x, c2.x, c3.x,
                    c0.y, c1.y, c2.y, c3.y,
                    c0.z, c1.z, c2.z, c3.z,
                    c0.w, c1.w, c2.w, c3.w);
}

float4x4 MatrixForGPUSkinning(float4 boneWeight, int4 boneIndices)
{
    float4x4 mat_x = CalcBoneMatrix(boneIndices.x);
    float4x4 mat_y = CalcBoneMatrix(boneIndices.y);
    float4x4 mat_z = CalcBoneMatrix(boneIndices.z);
    float4x4 mat_w = CalcBoneMatrix(boneIndices.w);
    mat_x = boneWeight.x * mat_x;
    mat_y = boneWeight.y * mat_y;
    mat_z = boneWeight.z * mat_z;
    mat_w = boneWeight.w * mat_w;

    return mat_x + mat_y + mat_z + mat_w;
}

[numthreads(GROUP_SIZE,1,1)]
void Skinning(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= asuint(_MeshLayout.w))
    {
        return;
    }

    int idx = id.x;
    int stride = asint(_MeshLayout.x);
    int vertexStart = stride * idx;
    float3 v0 = asfloat(_MeshVertice.Load3(vertexStart));
    

    // compute skinned position
    float4 skinnedPos = float4(v0, 1.0);

    int indexPack = asint(_MeshVertice.Load(vertexStart + asint(_MeshLayout.z)));
    int4 index = UnpackInt4FromInt32(indexPack);

    float4x4 mat = MatrixForGPUSkinning(asfloat(_MeshVertice.Load4(vertexStart + asint(_MeshLayout.y))), index);
    skinnedPos = mul(mat, skinnedPos);
    _MeshVerticeDst.Store3(vertexStart, asuint(skinnedPos.xyz));
}
#include "common.hlsl"

#include "gaussianDataTypes.hlsl"

cbuffer Constants : register(b0)
{
    float4x4 _MatrixWorldToObject;
    float4x4 _MatrixM;
    float4x4 _MatrixV[2];
    float4x4 _MatrixP[2];
    float4 _CameraPosWS[2];
    float4 _SplatProp;  // x: splat count, y: splat format, z: sh order, w: vertex stride
    float4 _VecScreenParams; // x: screen width
};

StructuredBuffer<int> _GSIndice : register(t1);
StructuredBuffer<int> _TriangleCullFlag : register(t2);
ByteAddressBuffer _SplatPropData : register(t3); //currently render face prop encoded as byte
StructuredBuffer<SplatChunkInfo> _SplatChunks : register(t4);
ByteAddressBuffer _SplatPos : register(t5);
ByteAddressBuffer _SplatOther : register(t6);
ByteAddressBuffer _SplatSH : register(t7);
Texture2D _SplatColor : register(t8);
StructuredBuffer<int> _MeshIndice : register(t9);
ByteAddressBuffer _MeshVertice : register(t10);
StructuredBuffer<float> _PoseShadowCompensation : register(t11);

RWStructuredBuffer<uint> _VisibleMasks : register(u12);
RWStructuredBuffer<float> _PointDistances : register(u13);
RWStructuredBuffer<SplatProjData> _SplatProjData0 : register(u14);
RWStructuredBuffer<SplatProjData> _SplatProjData1 : register(u15);

#include "gaussianSplatting.hlsl"

void ClipPoint(uint idx)
{
    _VisibleMasks[idx] = 0;
    _SplatProjData0[idx] = (SplatProjData)0;
#ifdef IS_STEREO_PIPELINE
    _SplatProjData1[idx] = (SplatProjData)0;
#endif
}

bool InFrustum(float4 clipPos)
{
    float w = clipPos.w;
    return clipPos.w > 0 && abs(clipPos.x) <= w && abs(clipPos.y) <= w && abs(clipPos.z) <= w;
}

SplatProjData CalcSplatProjData(SplatData splat, float3 centerWorldPos, float4 centerClipPos, float3 cov3d0, float3 cov3d1, uint shOrder, float3 camWorldPos, float4x4 matrixMV, float4x4 matrixP, float3x3 triTrans, float shadow_atten)
{
    // position
    SplatProjData view = (SplatProjData)0;
    view.pos = centerClipPos;

    // axis
    float4 cov2d = CalcCovariance2D(splat.pos, cov3d0, cov3d1, matrixMV, matrixP, _VecScreenParams);
    DecomposeCovariance(cov2d.xyz, view.axis1, view.axis2);

    // color
    float3 worldViewDir = camWorldPos.xyz - centerWorldPos;
    float3 objViewDir = mul(triTrans, mul((float3x3) _MatrixWorldToObject, worldViewDir));
    objViewDir.y = -objViewDir.y;
    objViewDir = normalize(objViewDir);
    
    half4 col;
    col.rgb = ShadeSH(splat.sh, objViewDir, shOrder);
    col.rgb *= shadow_atten;
    col.a = min(splat.opacity * cov2d.w, 65000);
    view.color = col;

    return view;
}

[numthreads(GROUP_SIZE,1,1)]
void CalcProjData (uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    if (idx >= GetSplatCount())
        return;

    int tri_id = _GSIndice[idx];
    int triFlag = _TriangleCullFlag[tri_id];

    uint gFrontFace = UnpackByteAsUint(_SplatPropData, idx);// 0 back, 1 front, 2 both
    if(gFrontFace != 2 && gFrontFace != uint(triFlag + 1))
    {
        ClipPoint(idx);
        return;
    }

    float3 pos = LoadSplatPos(idx);

    TriangleTransform trans = GetTriangleTransform(tri_id, pos);
    pos = trans.origin;
    float3x3 tri_mat = trans.jacobian;

    float3 centerWorldPos = mul(_MatrixM, float4(pos, 1)).xyz;
    // share viewPos between eyes
    float3 centerViewPos = mul(_MatrixV[0], float4(centerWorldPos, 1)).xyz;
    
    // cull points as it does in renderer used by training
    // camera forward is positive z
    if (centerViewPos.z <= 0.001)
    {
        ClipPoint(idx);
        return;
    }

    // frustum culling
    float4 centerClipPos[2];
    centerClipPos[0] = mul(_MatrixP[0], float4(centerViewPos, 1));
#ifdef IS_STEREO_PIPELINE
    bool visible[2];
    visible[0] = InFrustum(centerClipPos[0]);
    centerClipPos[1] = mul(mul(_MatrixP[1], _MatrixV[1]), float4(centerWorldPos, 1));
    visible[1] = InFrustum(centerClipPos[1]);
    if (!visible[0] && !visible[1])
#else  // !IS_STEREO_PIPELINE
    if (!InFrustum(centerClipPos[0]))
#endif  // IS_STEREO_PIPELINE
    {
        ClipPoint(idx);
        return;
    }

    // decode splat（much slowly）
    uint shOrder = GetSHOrder();
    SplatData splat = LoadSplatData(idx, shOrder);
    splat.rot = float4(-splat.rot.x, splat.rot.y, -splat.rot.z, splat.rot.w);

    // clip point with low opacity (as it does in training renderer)
    if (splat.opacity < 1.0 / 255.0)
    {
        ClipPoint(idx);
        return;
    }

    // mark point visible
    _VisibleMasks[idx] = 1;
    // assign object space position computed with triangle transform
    splat.pos = pos;
    float3x3 triTrans = transpose(trans.rotation);

    // calculate cov3D (view independent)
    float4 boxRot = splat.rot;
    float3 boxSize = splat.scale;
    float3x3 splatRotScaleMat = CalcMatrixFromRotationScale(boxRot, boxSize);
    splatRotScaleMat = mul(tri_mat, splatRotScaleMat);

    float3 cov3d0, cov3d1;
    CalcCovariance3D(splatRotScaleMat, cov3d0, cov3d1);
    float shadow_atten = trans.face_shadow;

    // record view space point distance for 'CalcPointDistance' step
    _PointDistances[idx] = centerViewPos.z;

#ifdef IS_STEREO_PIPELINE
    // calculate SplatProjData for left eye
    if (visible[0])
    {
        _SplatProjData0[idx] = CalcSplatProjData(splat, centerWorldPos, centerClipPos[0], cov3d0, cov3d1, shOrder, _CameraPosWS[0].xyz, mul(_MatrixV[0], _MatrixM), _MatrixP[0], triTrans, shadow_atten);
    }
    else
    {
        _SplatProjData0[idx] = (SplatProjData)0;
    }

    // calculate SplatProjData for right eye
    if (visible[1])
    {
        _SplatProjData1[idx] = CalcSplatProjData(splat, centerWorldPos, centerClipPos[1], cov3d0, cov3d1, shOrder, _CameraPosWS[1].xyz, mul(_MatrixV[1], _MatrixM), _MatrixP[1], triTrans, shadow_atten);
    }
    else
    {
        _SplatProjData1[idx] = (SplatProjData)0;
    }
#else
    _SplatProjData0[idx] = CalcSplatProjData(splat, centerWorldPos, centerClipPos[0], cov3d0, cov3d1, shOrder, _CameraPosWS[0].xyz, mul(_MatrixV[0], _MatrixM), _MatrixP[0], triTrans, shadow_atten);
#endif

}
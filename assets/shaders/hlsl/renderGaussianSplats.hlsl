#include "gaussianDataTypes.hlsl"

struct v2f
{
    float4 vertex : SV_POSITION;
    half4 col : COLOR0;
    half2 pos : TEXCOORD0;
#ifdef IS_STEREO_PIPELINE
	uint stereoTargetEyeIndexAsRTArrayIdx : SV_RenderTargetArrayIndex;
#endif
};

cbuffer Constants : register(b0)
{
    float4 _CustomScreenParams;    // xy: screen size
};

StructuredBuffer<SplatProjData> _SplatProjData0 : register(t1);
StructuredBuffer<SplatProjData> _SplatProjData1 : register(t2);

StructuredBuffer<uint> _OrderBuffer : register(t3);
StructuredBuffer<uint> _VisiblePointList : register(t4);

v2f Vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
	v2f o = (v2f)0;

#ifdef IS_STEREO_PIPELINE
	uint eyeIndex = instID & 0x01;
	o.stereoTargetEyeIndexAsRTArrayIdx = eyeIndex;
	instID = instID >> 1;
#endif

#if ENABLE_INDIRECT
    instID = _VisiblePointList[instID];
#else
	instID = _OrderBuffer[instID];
#endif

#ifdef IS_STEREO_PIPELINE
	SplatProjData view;
	if (eyeIndex == 0)
	{
		view = _SplatProjData0[instID];
	}
	else
	{
		view = _SplatProjData1[instID];
	}
#else
	SplatProjData view = _SplatProjData0[instID];
#endif
	float4 centerClipPos = view.pos;

#if !ENABLE_INDIRECT || IS_STEREO_PIPELINE
	float w = centerClipPos.w;
	bool behindCam = w <= 0;
	if (behindCam)
	{
		// discard invisible primitive
		o.vertex = asfloat(0x7fc00000);
		return o;
	}
#endif

    o.col = view.color;

	// v0:(-1,-1), v1:(1, -1), v2:(-1, 1), v3:(1, 1)
	uint idx = vtxID;
	float2 quadPos = float2(idx & 1, (idx >> 1) & 1) * 2.0 - 1.0;
	float quadSize = sqrt(log(1.0 / 255.0 / o.col.a) * -0.5);
	quadPos *= quadSize;
	o.pos = quadPos;

	float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2 / _CustomScreenParams.xy;
	o.vertex = centerClipPos;
	o.vertex.xy += deltaScreenPos * centerClipPos.w;
	o.vertex.z = (o.vertex.z + o.vertex.w) * 0.5;

	o.vertex.y *= -1.0f;

    return o;
}

half4 Frag(v2f i) : SV_Target
{
	half power = -dot(i.pos.xy, i.pos.xy);
	half alpha = exp(power);
	alpha = saturate(alpha * i.col.a);
	return half4(i.col.rgb * alpha, alpha);
}

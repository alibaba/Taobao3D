
#ifdef IS_STEREO_PIPELINE
    Texture2DArray<half4> _InTexture : register(t0);
#else
    Texture2D<half4> _InTexture : register(t0);
#endif

SamplerState _Sampler : register(s1);

struct v2f
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
#ifdef IS_STEREO_PIPELINE
	uint stereoTargetEyeIndexAsRTArrayIdx : SV_RenderTargetArrayIndex;
#endif
};

float GammaToLinear(float c)
{
    float linearRGBLo  = c / 12.92;
    float linearRGBHi  = pow((c + 0.055) / 1.055, 2.4);
    float linearRGB    = (c <= 0.04045) ? linearRGBLo : linearRGBHi;
    return linearRGB;
}

float3 GammaToLinear(float3 c)
{
    return float3(GammaToLinear(c.r), GammaToLinear(c.g), GammaToLinear(c.b));
}

v2f Vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
#ifdef IS_STEREO_PIPELINE
	uint eyeIndex = instID & 0x01;
	o.stereoTargetEyeIndexAsRTArrayIdx = eyeIndex;
	instID = instID >> 1;
#endif
	// v0:(-1,-1), v1:(1, -1), v2:(-1, 1), v3:(1, 1)
	uint idx = vtxID;
	float2 quadPos = float2(idx & 1, (idx >> 1) & 1) * 2.0 - 1.0;
    o.pos = float4(quadPos, 0.5, 1.0);
    // uv0:(0,1), uv1:(1,1), uv2:(0,0), uv3:(1,0)
    o.uv = float2((idx&1), (idx>>1)^1);

    return o;
}

half4 Frag(v2f i) : SV_Target
{
#ifdef IS_STEREO_PIPELINE
    half4 color = _InTexture.Sample(_Sampler, float3(i.uv.xy, i.stereoTargetEyeIndexAsRTArrayIdx));
#else
    half4 color = _InTexture.Sample(_Sampler, i.uv.xy);
#endif

    color.rgb = GammaToLinear(color.rgb);
    return color;
}
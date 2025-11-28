
cbuffer Constants : register(b0)
{
    float4x4 _MVP[2];
};

struct VertexInput
{
    float3 position : POSITION0;
};

struct VertexOutput
{
    float4 position : SV_Position;
#ifdef IS_STEREO_PIPELINE
	uint stereoTargetEyeIndexAsRTArrayIdx : SV_RenderTargetArrayIndex;
#endif
};

VertexOutput Vert (VertexInput input, uint instID : SV_InstanceID)
{
    VertexOutput o = (VertexOutput)0;

#ifdef IS_STEREO_PIPELINE
	uint eyeIndex = instID & 0x01;
	o.stereoTargetEyeIndexAsRTArrayIdx = eyeIndex;
#else
    uint eyeIndex = 0;
#endif
    o.position = mul(_MVP[eyeIndex], float4(input.position, 1.0f));

    return o;
}

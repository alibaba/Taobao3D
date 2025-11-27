#include "common.hlsl"

cbuffer Constants : register(b0)
{
    float4 _Params;  // x: texture width, y: texture height, z: pixel count, w: isFlipY
};

#ifdef IS_STEREO_PIPELINE
Texture2DArray<half4> _InTexture : register(t1);
RWTexture2DArray<half4> _OutTexture : register(u2);
#else
Texture2D<half4> _InTexture : register(t1);
RWTexture2D<half4> _OutTexture : register(u2);
#endif

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

[numthreads(GROUP_SIZE, 1, 1)]
void Postprocess(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= asuint(_Params.z))
        return;
    
    uint texWidth = asuint(_Params.x);
    uint width = id.x % texWidth;
    uint height = id.x / texWidth;
    // if (asuint(_Params.w))
    //     height = asuint(_Params.y) - height - 1;
#ifdef IS_STEREO_PIPELINE
    half4 color = _InTexture.Load(int4(width, height, id.z, 0));
#else
    half4 color = _InTexture.Load(int3(width, height, 0));
#endif
    color.rgb = GammaToLinear(color.rgb);

#ifdef IS_STEREO_PIPELINE
    _OutTexture[int3(width, height, id.z)] = color;
#else
    _OutTexture[int2(width, height)] = color;
#endif
}
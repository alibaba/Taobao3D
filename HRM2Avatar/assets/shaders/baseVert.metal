#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 pos [[position]];
    float2 uv;
};

struct Uniforms {
    float4x4 model;
    float4x4 view[2];
    float4x4 projection[2];
};

vertex VertexOut vertexMain(VertexIn in [[stage_in]],
                            constant Uniforms &u [[buffer(0)]])
{
    VertexOut out;
    float4 worldPos = u.model * float4(in.position, 1.0);
    float4 viewPos  = u.view[0] * worldPos;
    out.pos         = u.projection[0] * viewPos;
    out.uv          = in.texCoord;
    return out;
}
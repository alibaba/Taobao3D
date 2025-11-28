#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 pos [[position]];
    float2 uv;
};

fragment float4 fragmentMain(VertexOut in [[stage_in]])
{
    float2 c = floor(in.uv * 16.0);
    bool check = bool(int(c.x + c.y) & 1);
    return check ? float4(0.9, 0.9, 0.9, 1.0)
                 : float4(0.2, 0.2, 0.2, 1.0);
}
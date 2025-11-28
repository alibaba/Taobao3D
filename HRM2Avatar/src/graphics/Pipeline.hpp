#pragma once

#include <memory>

#include <Metal/Metal.hpp>

#include "Metal/MTLPixelFormat.hpp"
#include "Shader.hpp"

#define VERTEX_BUFFER_BINDING_OFFSET 16

namespace hrm
{

class ComputePipeline
{
public:
    ComputePipeline(MTL::Device* device, std::unique_ptr<Shader>&& shader);
    ~ComputePipeline();

    MTL::ComputePipelineState* GetPipelineState() const { return m_pPipelineState; }
    Shader* GetShader() const { return m_Shader.get(); }
    void SetShader(std::unique_ptr<Shader>&& shader) { m_Shader = std::move(shader); }

private:
    MTL::Device* m_pDevice { nullptr };
    MTL::ComputePipelineState* m_pPipelineState { nullptr };

    std::unique_ptr<Shader> m_Shader { nullptr };
};

struct VertexBufferLayoutDesc
{
    MTL::VertexStepFunction stepFunction { MTL::VertexStepFunctionConstant };
    uint32_t stride { 0u };
    uint32_t stepRate { 0u };
};

struct VertexAttributeDesc
{
    MTL::VertexFormat format { MTL::VertexFormatFloat };
    uint32_t offset { 0u };
    uint32_t bufferIndex { 0u };
    uint32_t location { 0u };
};

struct BlendStateDesc
{
    bool blendingEnabled { true };
    MTL::PixelFormat pixelFormat { MTL::PixelFormatRGBA16Float };
    MTL::ColorWriteMask writeMask { MTL::ColorWriteMaskAll };

    MTL::BlendOperation rgbBlendOperation { MTL::BlendOperationAdd };
    MTL::BlendOperation alphaBlendOperation { MTL::BlendOperationAdd };
    MTL::BlendFactor sourceRGBBlendFactor { MTL::BlendFactorDestinationColor };
    MTL::BlendFactor destinationRGBBlendFactor { MTL::BlendFactorOne };
    MTL::BlendFactor sourceAlphaBlendFactor { MTL::BlendFactorZero };
    MTL::BlendFactor destinationAlphaBlendFactor { MTL::BlendFactorOneMinusSource1Alpha };
};

struct RenderPipelineDesc
{
    MTL::PrimitiveTopologyClass primitiveType { MTL::PrimitiveTopologyClassTriangle };

    //------------------vertex buffer layout---------------------//
    std::vector<VertexBufferLayoutDesc> vertexBufferLayouts;
    std::vector<VertexAttributeDesc> vertexAttributes;

    //------------------depth stencil state----------------------//
    MTL::PixelFormat depthAttachmentPixelFormat { MTL::PixelFormatInvalid };
    bool depthWriteEnabled { false };
    MTL::CompareFunction depthCompareFunction { MTL::CompareFunctionAlways };
    // TODO: add stencil config
    bool stencilTestEnable { false };

    //------------------rasterization state----------------------//
    bool rasterizationEnabled { true };
    uint32_t sampleCount { 1u };
    MTL::TriangleFillMode triangleFillMode { MTL::TriangleFillModeFill };
    MTL::CullMode cullMode { MTL::CullModeNone };
    MTL::Winding frontFacingWinding { MTL::WindingCounterClockwise };

    //------------------blend state----------------------//
    std::vector<BlendStateDesc> blendStates;
};

class RenderPipeline
{
public:
    RenderPipeline(MTL::Device* device, const RenderPipelineDesc& desc, std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader = nullptr);
    ~RenderPipeline() = default;

    MTL::RenderPipelineState* GetPipelineState() const { return m_pPipelineState; }
    MTL::DepthStencilState* GetDepthStencilState() const { return m_pDepthStencilState; }

    const RenderPipelineDesc& GetDesc() const { return m_Desc; }
    RenderPipelineDesc& GetDesc() { return m_Desc; }

    Shader* GetVertexShader() const { return m_VertexShader.get(); }
    void SetVertexShader(std::unique_ptr<Shader>&& vertexShader) { m_VertexShader = std::move(vertexShader); }
    Shader* GetFragmentShader() const { return m_FragmentShader.get(); }
    void SetFragmentShader(std::unique_ptr<Shader>&& fragmentShader) { m_FragmentShader = std::move(fragmentShader); }

private:
    MTL::Device* m_pDevice { nullptr };
    MTL::RenderPipelineState* m_pPipelineState { nullptr };
    MTL::DepthStencilState*   m_pDepthStencilState { nullptr };

    RenderPipelineDesc m_Desc;
    std::unique_ptr<Shader> m_VertexShader { nullptr };
    std::unique_ptr<Shader> m_FragmentShader { nullptr };
};

}

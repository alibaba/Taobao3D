#include "Pipeline.hpp"

#include "Logging.hpp"

#define MAX_THREADS_PER_THREADGROUP 512

namespace hrm
{

ComputePipeline::ComputePipeline(MTL::Device* device, std::unique_ptr<Shader>&& shader)
    : m_pDevice(device), m_Shader(std::move(shader))
{
    assert(device != nullptr && m_Shader != nullptr);

    MTL::ComputePipelineDescriptor* pDescriptor = MTL::ComputePipelineDescriptor::alloc()->init();
    pDescriptor->setComputeFunction(m_Shader->GetHandle());
    pDescriptor->setLabel(NS::String::string(m_Shader->GetDesc().name.c_str(), NS::StringEncoding::UTF8StringEncoding));
    pDescriptor->setMaxTotalThreadsPerThreadgroup(MAX_THREADS_PER_THREADGROUP);

    NS::Error* pError = nullptr;
    m_pPipelineState = device->newComputePipelineState(pDescriptor, MTL::PipelineOptionNone, nullptr, &pError);
    if (pError != nullptr)
    {
        LOG_ERROR("Failed to create compute pipeline state: %s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pDescriptor->release();
}

ComputePipeline::~ComputePipeline()
{
    if (m_pPipelineState != nullptr)
    {
        m_pPipelineState->release();
    }
}

RenderPipeline::RenderPipeline(MTL::Device* device, const RenderPipelineDesc& desc, std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader)
    : m_pDevice(device), m_Desc(desc), m_VertexShader(std::move(vertexShader)), m_FragmentShader(std::move(fragmentShader))
{
    assert(device != nullptr && m_VertexShader != nullptr);

    MTL::RenderPipelineDescriptor* pDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    pDescriptor->setVertexFunction(m_VertexShader->GetHandle());
    if (m_FragmentShader)
        pDescriptor->setFragmentFunction(m_FragmentShader->GetHandle());
    pDescriptor->setLabel(NS::String::string(m_VertexShader->GetDesc().name.c_str(), NS::StringEncoding::UTF8StringEncoding));

    pDescriptor->setInputPrimitiveTopology(m_Desc.primitiveType);

    //---------------------vertex layout---------------------//
    MTL::VertexDescriptor* pVertexDescriptor = MTL::VertexDescriptor::alloc()->init();
    uint32_t bindingIndex = VERTEX_BUFFER_BINDING_OFFSET;
    for (const auto& bufferDesc : m_Desc.vertexBufferLayouts)
    {
        auto pBufferDescriptor = NS::TransferPtr(MTL::VertexBufferLayoutDescriptor::alloc()->init());
        pBufferDescriptor->setStepFunction(bufferDesc.stepFunction);
        pBufferDescriptor->setStepRate(bufferDesc.stepRate);
        pBufferDescriptor->setStride(bufferDesc.stride);
        pVertexDescriptor->layouts()->setObject(pBufferDescriptor.get(), bindingIndex++);
    }
    for (const auto& attribute : m_Desc.vertexAttributes)
    {
        auto pAttributeDescriptor = NS::TransferPtr(MTL::VertexAttributeDescriptor::alloc()->init());
        pAttributeDescriptor->setFormat(attribute.format);
        pAttributeDescriptor->setOffset(attribute.offset);
        pAttributeDescriptor->setBufferIndex(attribute.bufferIndex + VERTEX_BUFFER_BINDING_OFFSET);
        pVertexDescriptor->attributes()->setObject(pAttributeDescriptor.get(), attribute.location);
    }
    pDescriptor->setVertexDescriptor(pVertexDescriptor);

    //---------------------depth stencil state---------------------//
    pDescriptor->setDepthAttachmentPixelFormat(m_Desc.depthAttachmentPixelFormat);
    MTL::DepthStencilDescriptor* pDepthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init();
    pDepthStencilDescriptor->setDepthWriteEnabled(m_Desc.depthWriteEnabled);
    pDepthStencilDescriptor->setDepthCompareFunction(m_Desc.depthCompareFunction);
    m_pDepthStencilState = device->newDepthStencilState(pDepthStencilDescriptor);

    //---------------------rasterization state---------------------//
    pDescriptor->setRasterizationEnabled(m_Desc.rasterizationEnabled);
    pDescriptor->setRasterSampleCount(m_Desc.sampleCount);

    //---------------------blend state---------------------//
    uint32_t attachmentIndex = 0;
    for (const auto& blendState : m_Desc.blendStates)
    {
        auto pColorAttachmentDescriptor = NS::TransferPtr(MTL::RenderPipelineColorAttachmentDescriptor::alloc()->init());
        pColorAttachmentDescriptor->setBlendingEnabled(blendState.blendingEnabled);
        pColorAttachmentDescriptor->setPixelFormat(blendState.pixelFormat);
        pColorAttachmentDescriptor->setSourceRGBBlendFactor(blendState.sourceRGBBlendFactor);
        pColorAttachmentDescriptor->setDestinationRGBBlendFactor(blendState.destinationRGBBlendFactor);
        pColorAttachmentDescriptor->setSourceAlphaBlendFactor(blendState.sourceAlphaBlendFactor);
        pColorAttachmentDescriptor->setDestinationAlphaBlendFactor(blendState.destinationAlphaBlendFactor);
        pColorAttachmentDescriptor->setRgbBlendOperation(blendState.rgbBlendOperation);
        pColorAttachmentDescriptor->setAlphaBlendOperation(blendState.alphaBlendOperation);
        pColorAttachmentDescriptor->setWriteMask(blendState.writeMask);
        pDescriptor->colorAttachments()->setObject(pColorAttachmentDescriptor.get(), attachmentIndex++);
    }
    
    NS::Error* pError = nullptr;
    m_pPipelineState = device->newRenderPipelineState(pDescriptor, &pError);
    if (pError != nullptr)
    {
        LOG_ERROR("Failed to create render pipeline state: %s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pDescriptor->release();
    pVertexDescriptor->release();
    pDepthStencilDescriptor->release();
}

}  // namespace hrm

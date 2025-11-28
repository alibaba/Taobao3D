#include "Graphics.hpp"

#include "Logging.hpp"
#include "Drawable.hpp"
#include "Metal/MTLRenderPipeline.hpp"
#include "Metal/MTLSampler.hpp"
#include "Pipeline.hpp"
#include "Renderer.hpp"
#include "AssetLoader.hpp"
#include "Mesh.hpp"
#include "MetalHelper.hpp"
#include "ShaderTypes.hpp"

namespace hrm
{

Graphics::Graphics(Renderer* pRenderer)
    : m_pRenderer(pRenderer)
{
    m_pDevice = m_pRenderer->GetDevice();
    m_pCommandQueue = m_pDevice->newCommandQueue();

    m_PerFrameBufferPool = std::make_unique<PerFrameBufferPool>(pRenderer);
}

Graphics::~Graphics()
{
    if (m_pStagingBuffer != nullptr)
    {
        m_pStagingBuffer->release();
    }
    if (m_pCommandQueue != nullptr)
    {
        m_pCommandQueue->release();
    }
    if (m_pAutoreleasePool != nullptr)
    {
        m_pAutoreleasePool->release();
    }
    if (m_pRenderTarget != nullptr)
    {
        m_pRenderTarget->release();
    }
    if (m_pNearestSamplerState != nullptr)
    {
        m_pNearestSamplerState->release();
    }
}

MTL::Buffer* Graphics::CreateBuffer(uint32_t size, MemoryType memoryType)
{
    switch (memoryType)
    {
    case MemoryType::Shared:
        return m_pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    case MemoryType::Device:
        return m_pDevice->newBuffer(size, MTL::ResourceStorageModePrivate);
    default:
        LOG_ERROR("Unsupported memory type: %d", static_cast<int>(memoryType));
        return nullptr;
    }
}

void Graphics::CopyToDeviceBuffer(const uint8_t* data, uint32_t size, MTL::Buffer* deviceBuffer)
{
    UpdateStagingBuffer(size);

    memcpy(m_pStagingBuffer->contents(), data, size);
    MTL::CommandBuffer* pBlitCommandBuffer = m_pCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* pBlitCommandEncoder = pBlitCommandBuffer->blitCommandEncoder();
    pBlitCommandEncoder->copyFromBuffer(m_pStagingBuffer, 0, deviceBuffer, 0, size);
    pBlitCommandEncoder->endEncoding();

    pBlitCommandBuffer->commit();
    pBlitCommandBuffer->waitUntilCompleted();
}

void Graphics::CopyToDeviceTexture(const uint8_t* data, uint32_t size, MTL::Texture* deviceTexture)
{
    UpdateStagingBuffer(size);
    
    memcpy(m_pStagingBuffer->contents(), data, size);
    MTL::CommandBuffer* pBlitCommandBuffer = m_pCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* pBlitCommandEncoder = pBlitCommandBuffer->blitCommandEncoder();

    MTL::PixelFormat pixelFormat = deviceTexture->pixelFormat();
    uint32_t bpp = MetalHelper::GetBytesPerPixel(pixelFormat);
    uint32_t bytesPerRow = (uint32_t)deviceTexture->width() * bpp;
    uint32_t bytesPerSlice = (uint32_t)deviceTexture->height() * bytesPerRow;
    assert(bytesPerSlice == size);  // simplify the implementation, texture's level count is 1 and slice count is 1

    pBlitCommandEncoder->copyFromBuffer(m_pStagingBuffer,
                                        0,
                                        bytesPerRow,
                                        bytesPerSlice,
                                        MTL::Size(deviceTexture->width(), deviceTexture->height(), 1),
                                        deviceTexture,
                                        0,
                                        0,
                                        MTL::Origin(0, 0, 0));
    pBlitCommandEncoder->endEncoding();

    pBlitCommandBuffer->commit();
    pBlitCommandBuffer->waitUntilCompleted();
}

MTL::RenderCommandEncoder* Graphics::CreateGaussianRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex)
{
    if (pDrawable == nullptr)
        return nullptr;
    
    MTL::Texture* texture = pDrawable->GetTexture(viewIndex);
    if (texture == nullptr)
        return nullptr;
#ifdef PLATFORM_VISIONOS
    m_pRenderTarget = GetOrCreateRTForVisionOS(texture);
#else
    m_pRenderTarget = texture;
#endif
    
    MTL::RenderPassDescriptor* pRenderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    auto colorAtt = pRenderPassDescriptor->colorAttachments()->object(0);
    colorAtt->setTexture(m_pRenderTarget);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    colorAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));

    MTL::Texture* depthTexture = pDrawable->GetDepthTexture(viewIndex);
    if (depthTexture)
    {
        auto depthAtt = pRenderPassDescriptor->depthAttachment();
        depthAtt->setTexture(depthTexture);
        depthAtt->setLoadAction(MTL::LoadActionClear);
        depthAtt->setStoreAction(MTL::StoreActionStore);
        depthAtt->setClearDepth(1.0f);
    }

    if (MTL::RasterizationRateMap* rrm = pDrawable->GetRasterizationRateMap(viewIndex); rrm != nullptr)
    {
        pRenderPassDescriptor->setRasterizationRateMap(rrm);
    }

#ifndef PLATFORM_IOS
    pRenderPassDescriptor->setRenderTargetArrayLength(m_pRenderTarget->arrayLength());
#endif
    
    MTL::RenderCommandEncoder* encoder = m_pCommandBuffer->renderCommandEncoder(pRenderPassDescriptor);
    encoder->setLabel(NS::String::string("Render Command Encoder", NS::StringEncoding::UTF8StringEncoding));

    return encoder;
}

MTL::RenderCommandEncoder* Graphics::CreatePostprocessRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex)
{
    if (pDrawable == nullptr)
        return nullptr;
    MTL::Texture* texture = pDrawable->GetTexture(viewIndex);
    if (texture == nullptr)
        return nullptr;
    
    MTL::RenderPassDescriptor* pRenderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    auto colorAtt = pRenderPassDescriptor->colorAttachments()->object(0);
    colorAtt->setTexture(texture);
    colorAtt->setLoadAction(MTL::LoadActionLoad);
    colorAtt->setStoreAction(MTL::StoreActionStore);

    pRenderPassDescriptor->setRenderTargetArrayLength(texture->arrayLength());
    
    MTL::RenderCommandEncoder* encoder = m_pCommandBuffer->renderCommandEncoder(pRenderPassDescriptor);
    encoder->setLabel(NS::String::string("Postprocess Render Command Encoder", NS::StringEncoding::UTF8StringEncoding));

    return encoder;
}

MTL::RenderCommandEncoder* Graphics::CreateDrawDepthRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex)
{
    if (pDrawable == nullptr)
        return nullptr;
    MTL::Texture* depthRT = pDrawable->GetDepthTexture(viewIndex);
    if(depthRT == nullptr)
        return nullptr;
    
    MTL::RenderPassDescriptor* pRenderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    
    auto depthAtt = pRenderPassDescriptor->depthAttachment();
    depthAtt->setTexture(depthRT);
    depthAtt->setLoadAction(MTL::LoadActionLoad);
    depthAtt->setStoreAction(MTL::StoreActionStore);

#ifndef PLATFORM_IOS
    pRenderPassDescriptor->setRenderTargetArrayLength(depthRT->arrayLength());
#endif
    
    MTL::RenderCommandEncoder* encoder = m_pCommandBuffer->renderCommandEncoder(pRenderPassDescriptor);
    encoder->setLabel(NS::String::string("Draw Depth Render Command Encoder", NS::StringEncoding::UTF8StringEncoding));
    return encoder;
}

void Graphics::PresentDrawable(Drawable* pDrawable)
{
    if (pDrawable == nullptr)
        return;
    
#ifdef PLATFORM_VISIONOS
    auto handle = static_cast<MetalVisionDrawable*>(pDrawable)->GetHandle();
    MetalHelper::PresentWithDrawable(handle, m_pCommandBuffer);
#else
    MTL::Drawable* mtlDrawable = static_cast<MetalDrawable*>(pDrawable)->GetHandle();
    m_pCommandBuffer->presentDrawable(mtlDrawable);
#endif
}

void Graphics::CommitCommandBuffer()
{
    m_pCommandBuffer->addCompletedHandler([this](MTL::CommandBuffer* pCommandBuffer) {
        m_PerFrameBufferPool->RecycleFrameBuffer();
    });
    m_pCommandBuffer->commit();
}

void Graphics::BeginFrame()
{
    m_pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    m_pCommandBuffer = m_pCommandQueue->commandBuffer();

    m_PerFrameBufferPool->AllocateConstantsBuffer();
}

void Graphics::EndFrame()
{
    m_pAutoreleasePool->release();
    m_pAutoreleasePool = nullptr;
}

std::unique_ptr<RenderPipeline> Graphics::CreateTestRenderPipeline(Mesh* pMesh)
{
    if (m_pRenderer == nullptr || pMesh == nullptr)
        return nullptr;

    if (pMesh->GetVertexLayout().empty())
        return nullptr;

    RenderPipelineDesc desc;
    auto vertexShader = AssetLoader::LoadShader(m_pDevice, "shaders/baseVert.metal", "vertexMain");
    auto fragmentShader = AssetLoader::LoadShader(m_pDevice, "shaders/baseFrag.metal", "fragmentMain");
    const auto& vertexLayout = pMesh->GetVertexLayout();
    desc.vertexBufferLayouts.push_back(VertexBufferLayoutDesc{ .stepFunction = MTL::VertexStepFunctionPerVertex, .stride = m_pRenderer->GetMesh()->GetVertexStride(), .stepRate = 1 });
    for (uint32_t i = 0; i < vertexLayout.size(); ++i)
    {
        const auto& layout = vertexLayout[i];
        desc.vertexAttributes.push_back(VertexAttributeDesc{ .format = layout.format, .offset = layout.offsetBytes, .bufferIndex = 0, .location = i });
    }

    desc.blendStates.push_back(BlendStateDesc{ .blendingEnabled = false });
#ifdef PLATFORM_VISIONOS
//    desc.depthAttachmentPixelFormat = MTL::PixelFormatDepth32Float;
//    desc.depthWriteEnabled = true;
//    desc.depthCompareFunction = MTL::CompareFunctionLessEqual;
#endif
    
    return std::make_unique<RenderPipeline>(m_pDevice, desc, std::move(vertexShader), std::move(fragmentShader));
}

std::unique_ptr<RenderPipeline> Graphics::CreateGaussianRenderPipeline(std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader)
{
    assert(vertexShader != nullptr && fragmentShader != nullptr);
    RenderPipelineDesc desc;
    desc.vertexBufferLayouts.push_back(VertexBufferLayoutDesc{
        .stepFunction = MTL::VertexStepFunctionPerVertex,
        .stride = 0,  // don't need any vertex attributes
        .stepRate = 1
    });
    desc.blendStates.push_back(BlendStateDesc{
//        .pixelFormat = MTL::PixelFormatRGBA8Unorm_sRGB,
        .sourceRGBBlendFactor = MTL::BlendFactorOneMinusDestinationAlpha,
        .destinationRGBBlendFactor = MTL::BlendFactorOne,
        .sourceAlphaBlendFactor = MTL::BlendFactorOneMinusDestinationAlpha,
        .destinationAlphaBlendFactor = MTL::BlendFactorOne
    });
#ifdef PLATFORM_VISIONOS
    desc.depthAttachmentPixelFormat = MTL::PixelFormatDepth32Float;
#endif
    return std::make_unique<RenderPipeline>(m_pDevice, desc, std::move(vertexShader), std::move(fragmentShader));
}

std::unique_ptr<RenderPipeline> Graphics::CreteDepthRenderPipeline(std::unique_ptr<Shader>&& vertexShader)
{
    assert(vertexShader != nullptr);
    RenderPipelineDesc desc;
    desc.vertexBufferLayouts.push_back(VertexBufferLayoutDesc{
        .stepFunction = MTL::VertexStepFunctionPerVertex,
        .stride = m_pRenderer->GetMesh()->GetVertexStride(),
        .stepRate = 1
    });
    desc.vertexAttributes.push_back(VertexAttributeDesc{
        .format = MTL::VertexFormatFloat3,
        .offset = 0,  // position offset must be 0
        .bufferIndex = 0,
        .location = 0
    });
    desc.blendStates.push_back(BlendStateDesc{
        .blendingEnabled = false,
        .pixelFormat = MTL::PixelFormatInvalid
    });
    desc.cullMode = MTL::CullModeBack;
#ifdef PLATFORM_VISIONOS
    desc.depthAttachmentPixelFormat = MTL::PixelFormatDepth32Float;
    desc.depthWriteEnabled = true;
    desc.depthCompareFunction = MTL::CompareFunctionLessEqual;
#endif
    return std::make_unique<RenderPipeline>(m_pDevice, desc, std::move(vertexShader));
}

std::unique_ptr<RenderPipeline> Graphics::CreatePostprocessRenderPipeline(std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader)
{
    assert(vertexShader != nullptr && fragmentShader != nullptr);
    RenderPipelineDesc desc;
    desc.vertexBufferLayouts.push_back(VertexBufferLayoutDesc{
        .stepFunction = MTL::VertexStepFunctionPerVertex,
        .stride = 0,  // don't need any vertex attributes
        .stepRate = 1
    });
    desc.blendStates.push_back(BlendStateDesc{
        .blendingEnabled = false,
        .pixelFormat = MTL::PixelFormatRGBA16Float
    });

    return std::make_unique<RenderPipeline>(m_pDevice, desc, std::move(vertexShader), std::move(fragmentShader));
}

void Graphics::SetPipelineStateToEncoder(MTL::RenderCommandEncoder* pEncoder, RenderPipeline* pPipeline)
{
    assert(pEncoder != nullptr && pPipeline != nullptr);
    pEncoder->setRenderPipelineState(pPipeline->GetPipelineState());
    pEncoder->setDepthStencilState(pPipeline->GetDepthStencilState());
    
    pEncoder->setCullMode(pPipeline->GetDesc().cullMode);
    pEncoder->setFrontFacingWinding(pPipeline->GetDesc().frontFacingWinding);
    pEncoder->setTriangleFillMode(pPipeline->GetDesc().triangleFillMode);
}

void Graphics::SetPipelineStateToEncoder(MTL::ComputeCommandEncoder* pEncoder, ComputePipeline* pPipeline)
{
    assert(pEncoder != nullptr && pPipeline != nullptr);
    pEncoder->setComputePipelineState(pPipeline->GetPipelineState());
}

void Graphics::DrawMesh(MTL::RenderCommandEncoder* pEncoder, Mesh* pMesh, uint32_t instanceCount, Buffer* indirectBuffer)
{
    assert(pEncoder != nullptr && pMesh != nullptr);
    auto vertexBuffer = pMesh->GetVertexBuffer();
    if (vertexBuffer != nullptr)
    {
        pEncoder->setVertexBuffer(vertexBuffer->GetHandle(), 0, VERTEX_BUFFER_BINDING_OFFSET);
    }
#if IS_STEREO_PIPELINE
    instanceCount *= 2;
#endif
    if (indirectBuffer != nullptr)
    {
        pEncoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                                        MTL::IndexType::IndexTypeUInt32,
                                        pMesh->GetIndexBuffer()->GetHandle(),
                                        0,
                                        indirectBuffer->GetHandle(),
                                        0);
    }
    else
    {
        pEncoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                                        pMesh->GetIndexCount(),
                                        MTL::IndexType::IndexTypeUInt32,
                                        pMesh->GetIndexBuffer()->GetHandle(),
                                        0,
                                        instanceCount);
    }
}

MTL::Texture* Graphics::CreateSimpleTexture2D(uint32_t width, uint32_t height, MTL::PixelFormat pixelFormat)
{
    MTL::TextureDescriptor* pTextureDescriptor = MTL::TextureDescriptor::texture2DDescriptor(pixelFormat, width, height, false);
    pTextureDescriptor->setResourceOptions(MTL::ResourceStorageModePrivate);
    pTextureDescriptor->setUsage(MTL::TextureUsageShaderRead);
    return m_pDevice->newTexture(pTextureDescriptor);
}

void Graphics::UpdateStagingBuffer(uint32_t size)
{
    if (m_pStagingBuffer == nullptr)
    {
        m_pStagingBuffer = m_pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    }
    else if (m_pStagingBuffer->length() < size)
    {
        m_pStagingBuffer->release();
        m_pStagingBuffer = m_pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    }
    m_pStagingBuffer->setLabel(NS::String::string("StagingBuffer", NS::StringEncoding::UTF8StringEncoding));
}

MTL::Texture* Graphics::GetOrCreateRTForVisionOS(MTL::Texture* texture)
{
    assert(texture != nullptr);
    if (m_pRenderTarget == nullptr)
    {
        MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, texture->width(), texture->height(), false);
        pTextureDesc->setResourceOptions(MTL::ResourceStorageModePrivate);
        pTextureDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        pTextureDesc->setTextureType(texture->textureType());
        pTextureDesc->setArrayLength(texture->arrayLength());
        m_pRenderTarget = m_pDevice->newTexture(pTextureDesc);
        m_pRenderTarget->setLabel(NS::String::string("VisionOS RT", NS::StringEncoding::UTF8StringEncoding));
    }
    return m_pRenderTarget;
}

MTL::SamplerState* Graphics::GetOrCreateNearestSamplerState()
{
    if (m_pNearestSamplerState == nullptr)
    {
        MTL::SamplerDescriptor* pSamplerDesc = MTL::SamplerDescriptor::alloc()->init();
        pSamplerDesc->setMinFilter(MTL::SamplerMinMagFilterNearest);
        pSamplerDesc->setMagFilter(MTL::SamplerMinMagFilterNearest);
        pSamplerDesc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
        m_pNearestSamplerState = m_pDevice->newSamplerState(pSamplerDesc);
        pSamplerDesc->setLabel(NS::String::string("Nearest Sampler", NS::StringEncoding::UTF8StringEncoding));
        pSamplerDesc->release();
    }
    return m_pNearestSamplerState;
}

}  // namespace hrm

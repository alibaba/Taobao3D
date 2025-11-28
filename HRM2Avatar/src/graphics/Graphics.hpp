#pragma once
        
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include "Pipeline.hpp"
#include "PerFrameBufferPool.hpp"

namespace hrm
{

class Drawable;
class Renderer;
class Mesh;

class Graphics
{

public:
    Graphics(Renderer* pRenderer);
    ~Graphics();

    MTL::Buffer* CreateBuffer(uint32_t size, MemoryType memoryType);
    
    void CopyToDeviceBuffer(const uint8_t* data, uint32_t size, MTL::Buffer* deviceBuffer);
    void CopyToDeviceTexture(const uint8_t* data, uint32_t size, MTL::Texture* deviceTexture);

    void FlushTransferCommands();

    // TODO: create render command from custom render pass descriptor
    MTL::RenderCommandEncoder* CreateGaussianRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex);
    MTL::RenderCommandEncoder* CreatePostprocessRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex);
    MTL::RenderCommandEncoder* CreateDrawDepthRenderCommandEncoder(Drawable* pDrawable, uint32_t viewIndex);

    MTL::ComputeCommandEncoder* CreateComputeCommandEncoder() { return m_pCommandBuffer->computeCommandEncoder(); }
    
    void BeginFrame();

    void EndFrame();

    void PresentDrawable(Drawable* pDrawable);

    void CommitCommandBuffer();

    std::unique_ptr<RenderPipeline> CreateTestRenderPipeline(Mesh* pMesh);

    std::unique_ptr<RenderPipeline> CreateGaussianRenderPipeline(std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader);

    std::unique_ptr<RenderPipeline> CreatePostprocessRenderPipeline(std::unique_ptr<Shader>&& vertexShader, std::unique_ptr<Shader>&& fragmentShader);
    
    std::unique_ptr<RenderPipeline> CreteDepthRenderPipeline(std::unique_ptr<Shader>&& vertexShader); // depth only

    void SetPipelineStateToEncoder(MTL::RenderCommandEncoder* pEncoder, RenderPipeline* pPipeline);

    void SetPipelineStateToEncoder(MTL::ComputeCommandEncoder* pEncoder, ComputePipeline* pPipeline);

    void DrawMesh(MTL::RenderCommandEncoder* pEncoder, Mesh* pMesh, uint32_t instanceCount = 1, Buffer* indirectBuffer = nullptr);

    Buffer* AllocatePerFrameSkinningBuffer() { return m_PerFrameBufferPool->AllocateSkinningBuffer();}
    Buffer* AllocatePerFrameVertexBuffer() { return m_PerFrameBufferPool->AllocateVertexBuffer(); }
    Buffer* AllocatePerFrameConstantsBuffer(uint32_t size, uint32_t& offset) { return m_PerFrameBufferPool->TryAllocateConstantsBuffer(size, offset); }
    Buffer* AllocatePerFramePoseShadowCompensationBuffer() { return m_PerFrameBufferPool->AllocatePoseShadowCompensationBuffer(); }

    MTL::Texture* CreateSimpleTexture2D(uint32_t width, uint32_t height, MTL::PixelFormat pixelFormat);

    MTL::Texture* GetRenderTarget() const { return m_pRenderTarget; }

    MTL::SamplerState* GetOrCreateNearestSamplerState();

private:
    void UpdateStagingBuffer(uint32_t size);

    MTL::Texture* GetOrCreateRTForVisionOS(MTL::Texture* texture);

private:
    Renderer* m_pRenderer { nullptr };
    
    MTL::Device*        m_pDevice { nullptr };
    MTL::CommandQueue*  m_pCommandQueue { nullptr };
    MTL::CommandBuffer* m_pCommandBuffer { nullptr };

    MTL::Buffer* m_pStagingBuffer { nullptr };

    NS::AutoreleasePool* m_pAutoreleasePool { nullptr };

    std::unique_ptr<PerFrameBufferPool> m_PerFrameBufferPool { nullptr };

    std::unique_ptr<RenderPipeline> m_TestRenderPipeline { nullptr };  // used for testing

    MTL::Texture* m_pRenderTarget { nullptr};
    MTL::SamplerState* m_pNearestSamplerState { nullptr };
};

}  // namespace hrm

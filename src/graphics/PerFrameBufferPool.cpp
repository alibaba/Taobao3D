#include "PerFrameBufferPool.hpp"

#include "Renderer.hpp"
#include "ShaderTypes.hpp"

#define CONSTANTS_BUFFER_SIZE 8192  // 8KB = 32 * 256
#define ALIGNMENT_SIZE 256

namespace hrm
{

std::mutex g_ThreadMutex;

Buffer* PerFrameBufferPool::AllocateVertexBuffer()
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    if (!m_ReleasedVertexBuffers.empty())
    {
        m_UsedVertexBuffers.push_back(std::move(m_ReleasedVertexBuffers.front()));
        m_ReleasedVertexBuffers.pop_front();
    }
    else
    {
        uint32_t size = m_pRenderer->GetMesh()->GetVertexBuffer()->GetSize();
        std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(m_pRenderer->GetGraphics(), size, MemoryType::Shared);
        buffer->SetName("per frame VertexBuffer");
        m_UsedVertexBuffers.push_back(std::move(buffer));
    }
    return m_UsedVertexBuffers.back().get();
}

Buffer* PerFrameBufferPool::AllocateConstantsBuffer()
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    if (!m_ReleasedConstantsBuffers.empty())
    {
        m_UsedConstantsBuffers.push_back(std::move(m_ReleasedConstantsBuffers.front()));
        m_ReleasedConstantsBuffers.pop_front();
    }
    else
    {
        uint32_t size = CONSTANTS_BUFFER_SIZE;
        std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(m_pRenderer->GetGraphics(), size, MemoryType::Shared);
        buffer->SetName("per frame ConstantsBuffer");
        m_UsedConstantsBuffers.push_back(std::move(buffer));
    }
    m_ConstantsBufferOffset = 0u;
    return m_UsedConstantsBuffers.back().get();
}

Buffer* PerFrameBufferPool::TryAllocateConstantsBuffer(uint32_t size, uint32_t& offset)
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    size = (size + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE;
    assert(m_ConstantsBufferOffset + size <= CONSTANTS_BUFFER_SIZE);
    offset = m_ConstantsBufferOffset;
    m_ConstantsBufferOffset += size;
    return m_UsedConstantsBuffers.back().get();
}

Buffer* PerFrameBufferPool::AllocateSkinningBuffer()
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    if (!m_ReleasedSkinningBuffers.empty())
    {
        m_UsedSkinningBuffers.push_back(std::move(m_ReleasedSkinningBuffers.front()));
        m_ReleasedSkinningBuffers.pop_front();
    }
    else
    {
        uint32_t size = sizeof(SkinningConstants);
        std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(m_pRenderer->GetGraphics(), size, MemoryType::Shared);
        buffer->SetName("per frame SkinningConstants");
        m_UsedSkinningBuffers.push_back(std::move(buffer));
    }
    return m_UsedSkinningBuffers.back().get();
}

Buffer* PerFrameBufferPool::AllocatePoseShadowCompensationBuffer()
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    if (!m_ReleasedPoseShadowCompensationBuffers.empty())
    {
        m_UsedPoseShadowCompensationBuffers.push_back(std::move(m_ReleasedPoseShadowCompensationBuffers.front()));
        m_ReleasedPoseShadowCompensationBuffers.pop_front();
    }
    else
    {
        uint32_t size = m_pRenderer->GetMesh()->GetVertexCount() * sizeof(float);
        std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(m_pRenderer->GetGraphics(), size, MemoryType::Shared);
        buffer->SetName("per frame CompensationBuffer");
        m_UsedPoseShadowCompensationBuffers.push_back(std::move(buffer));
    }
    return m_UsedPoseShadowCompensationBuffers.back().get();
}

void PerFrameBufferPool::RecycleFrameBuffer()
{
    std::lock_guard<std::mutex> lock(g_ThreadMutex);
    if (!m_UsedVertexBuffers.empty())
    {
        m_ReleasedVertexBuffers.push_back(std::move(m_UsedVertexBuffers.front()));
        m_UsedVertexBuffers.pop_front();
    }
    if (!m_UsedConstantsBuffers.empty())
    {
        m_ReleasedConstantsBuffers.push_back(std::move(m_UsedConstantsBuffers.front()));
        m_UsedConstantsBuffers.pop_front();
    }
    if (!m_UsedSkinningBuffers.empty())
    {
        m_ReleasedSkinningBuffers.push_back(std::move(m_UsedSkinningBuffers.front()));
        m_UsedSkinningBuffers.pop_front();
    }
    if (!m_UsedPoseShadowCompensationBuffers.empty())
    {
        m_ReleasedPoseShadowCompensationBuffers.push_back(std::move(m_UsedPoseShadowCompensationBuffers.front()));
        m_UsedPoseShadowCompensationBuffers.pop_front();
    }
}

}

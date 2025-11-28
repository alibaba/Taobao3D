#pragma once

#include <deque>

#include "Buffer.hpp"

namespace hrm
{

class Renderer;

class PerFrameBufferPool
{
public:
    PerFrameBufferPool(Renderer* pRenderer) : m_pRenderer(pRenderer) {}
    ~PerFrameBufferPool() = default;

    Buffer* AllocateVertexBuffer();

    Buffer* AllocateConstantsBuffer();

    Buffer* TryAllocateConstantsBuffer(uint32_t size, uint32_t& offset);

    Buffer* AllocateSkinningBuffer();

    Buffer* AllocatePoseShadowCompensationBuffer();

    void RecycleFrameBuffer();

private:
    Renderer* m_pRenderer { nullptr };

    // NOTE: constants buffer size and vertex buffer size are fixed, could not be changed during runtime
    std::deque<std::unique_ptr<Buffer>> m_UsedVertexBuffers;
    std::deque<std::unique_ptr<Buffer>> m_ReleasedVertexBuffers;

    // 4KB per frame constants buffer
    std::deque<std::unique_ptr<Buffer>> m_UsedConstantsBuffers;
    std::deque<std::unique_ptr<Buffer>> m_ReleasedConstantsBuffers;

    std::deque<std::unique_ptr<Buffer>> m_UsedSkinningBuffers;
    std::deque<std::unique_ptr<Buffer>> m_ReleasedSkinningBuffers;

    std::deque<std::unique_ptr<Buffer>> m_UsedPoseShadowCompensationBuffers;
    std::deque<std::unique_ptr<Buffer>> m_ReleasedPoseShadowCompensationBuffers;

    uint32_t m_ConstantsBufferOffset { 0u };  // offset in constants buffer, reset to 0 every frame
};

}
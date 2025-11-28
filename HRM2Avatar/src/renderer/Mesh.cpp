#include "Mesh.hpp"

#include <cassert>

namespace hrm
{

void Mesh::SetVertexBufferData(uint8_t* vertexBuffer, uint32_t size)
{
    m_VertexBuffer = std::make_unique<Buffer>(m_pGraphics, size, MemoryType::Shared);
    m_VertexBuffer->SetData(vertexBuffer, size);
}

void Mesh::SetIndexBufferData(uint8_t* indexBuffer, uint32_t size)
{
    m_IndexBuffer = std::make_unique<Buffer>(m_pGraphics, size, MemoryType::Device);
    m_IndexBuffer->SetData(indexBuffer, size);
}

uint32_t Mesh::GetVertexCount() const
{
    assert(!m_VertexLayout.empty() && m_VertexLayout[0].strideBytes != 0);
    return (uint32_t)m_VertexBuffer->GetSize() / m_VertexLayout[0].strideBytes;
}

uint32_t Mesh::GetTriangleCount() const
{
    return (uint32_t)m_IndexBuffer->GetSize() / 12u;
}

} // namespace hrm

#pragma once

#include <vector>
#include <memory>

#include <Metal/Metal.hpp>

#include "Buffer.hpp"
#include "Metal/MTLVertexDescriptor.hpp"

namespace hrm
{

class Graphics;

enum class VertexAttributeType : uint8_t
{
    Position,
    Normal,
    Tangent,
    Bitangent,
    Color0,
    Color1,
    Color2,
    Color3,
    BoneWeight,
    BoneIndice,
    Texcoord0,
    Texcoord1,
    Texcoord2,
    Texcoord3,
    Texcoord4,
    Texcoord5,
    Texcoord6,
    Texcoord7,
    Count,
};

struct VertexLayout
{
    VertexAttributeType attribute { VertexAttributeType::Count };
    MTL::VertexFormat format { MTL::VertexFormatFloat };
    
    uint32_t offsetBytes { 0u };
    uint32_t strideBytes { 0u };
    uint32_t numElements { 0u };

    // TODO: modify this function
    static MTL::VertexFormat ConvertToMTLVertexFormat(VertexAttributeType elementType)
    {
        switch (elementType)
        {
            case VertexAttributeType::Position:
            case VertexAttributeType::Normal:
            case VertexAttributeType::Tangent:
            case VertexAttributeType::Bitangent:
                return MTL::VertexFormatFloat3;
            case VertexAttributeType::Color0:
            case VertexAttributeType::Color1:
            case VertexAttributeType::Color2:
                return MTL::VertexFormatUChar4Normalized;
            case VertexAttributeType::BoneWeight:
                return MTL::VertexFormatFloat4;
            case VertexAttributeType::BoneIndice:
                return MTL::VertexFormatUChar4;
            default:
                return MTL::VertexFormatInvalid;
        }
    }
};

class Mesh
{
public:
    Mesh(Graphics* pGraphics) : m_pGraphics(pGraphics) {}
    ~Mesh() = default;

    void SetVertexBufferData(uint8_t* vertexBuffer, uint32_t size);
    Buffer* GetVertexBuffer() const { return m_VertexBuffer.get(); }

    void SetIndexBufferData(uint8_t* indexBuffer, uint32_t size);
    Buffer* GetIndexBuffer() const { return m_IndexBuffer.get(); }

    void SetVertexLayout(std::vector<VertexLayout>&& vertexLayout) { m_VertexLayout = std::move(vertexLayout); }
    void SetVertexLayout(const std::vector<VertexLayout>& vertexLayout) { m_VertexLayout = vertexLayout; }
    const std::vector<VertexLayout>& GetVertexLayout() const { return m_VertexLayout; }

    uint32_t GetVertexStride() const { return m_VertexLayout.at(0).strideBytes; }
    
    uint32_t GetIndexCount() const { return m_IndexBuffer->GetSize() / sizeof(uint32_t); }
    uint32_t GetVertexCount() const;
    uint32_t GetTriangleCount() const;
    
private:
    Graphics* m_pGraphics { nullptr };

    std::unique_ptr<Buffer> m_VertexBuffer;
    std::unique_ptr<Buffer> m_IndexBuffer;
    std::vector<VertexLayout> m_VertexLayout;
};
    
}

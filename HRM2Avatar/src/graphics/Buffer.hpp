#pragma once

#include <vector>

#include <Metal/Metal.hpp>

namespace hrm
{

class Graphics;

enum class MemoryType : uint8_t
{
    Shared,
    Device,
    Count
};

class Buffer
{
public:
    Buffer(Graphics* pGraphics, uint32_t size, MemoryType memoryType);
    ~Buffer();

    void SetData(const uint8_t* data, uint32_t size);

    const uint8_t* GetData() const;
    uint8_t* GetDataToModify(uint32_t offset = 0u);

    uint32_t GetSize() const { return (uint32_t)m_pHandle->length(); }

    void SetName(const std::string& name) { m_Name = name; }
    const std::string& GetName() const { return m_Name; }

    MTL::Buffer* GetHandle();

private:
    Graphics*    m_pGraphics { nullptr };
    MTL::Buffer* m_pHandle { nullptr };

    std::vector<uint8_t> m_Data;  // only used for device memory type
    MemoryType           m_MemoryType { MemoryType::Count };

    bool m_IsDirty { false };
    std::string m_Name;
};

}  // namespace hrm

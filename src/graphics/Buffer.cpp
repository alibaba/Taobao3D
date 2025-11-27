#include "Buffer.hpp"

#include "Logging.hpp"
#include "Graphics.hpp"
#include "Renderer.hpp"

namespace hrm
{

Buffer::Buffer(Graphics* pGraphics, uint32_t size, MemoryType memoryType)
    : m_pGraphics(pGraphics), m_MemoryType(memoryType)
{
    m_pHandle = pGraphics->CreateBuffer(size, memoryType);
    assert(m_pHandle != nullptr);
    if (memoryType == MemoryType::Device)
    {
        m_Data.resize(size, 0x00);
        m_IsDirty = true;
    }
    else
    {
        memset(m_pHandle->contents(), 0x00, size);
    }
}

Buffer::~Buffer()
{
    if (m_pHandle != nullptr)
    {
        m_pHandle->release();
    }
}

void Buffer::SetData(const uint8_t* data, uint32_t size)
{
    assert(size == m_pHandle->length());
    switch (m_MemoryType)
    {
    case MemoryType::Shared:
        memcpy(m_pHandle->contents(), data, size);
        break;
    case MemoryType::Device:
    {
        m_Data.resize(size);
        memcpy(m_Data.data(), data, size);
        m_IsDirty = true;
        break;
    }
    default:
        LOG_ERROR("Unsupported memory type: %d", static_cast<int>(m_MemoryType));
        break;
    }
}

const uint8_t* Buffer::GetData() const
{
    switch (m_MemoryType)
    {
    case MemoryType::Shared:
        return (const uint8_t*)m_pHandle->contents();
    case MemoryType::Device:
        return (const uint8_t*)m_Data.data();
    default:
        LOG_ERROR("Unsupported memory type: %d", static_cast<int>(m_MemoryType));
        return nullptr;
    }
}

uint8_t* Buffer::GetDataToModify(uint32_t offset)
{
    m_IsDirty = true;
    switch (m_MemoryType)
    {
    case MemoryType::Shared:
        return (uint8_t*)(m_pHandle->contents()) + offset;
    case MemoryType::Device:
        return m_Data.data() + offset;
    default:
        LOG_ERROR("Unsupported memory type: %d", static_cast<int>(m_MemoryType));
        return nullptr;
    }
}

MTL::Buffer* Buffer::GetHandle()
{
    if (m_IsDirty && m_MemoryType == MemoryType::Device)
    {
        m_pGraphics->CopyToDeviceBuffer(m_Data.data(), (uint32_t)m_Data.size(), m_pHandle);
        m_IsDirty = false;
    }
    if (!m_Name.empty())
    {
        m_pHandle->setLabel(NS::String::string(m_Name.c_str(), NS::StringEncoding::UTF8StringEncoding));
    }
    return m_pHandle;
}

}  // namespace hrm

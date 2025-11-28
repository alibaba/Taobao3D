#include "Shader.hpp"

#include "Logging.hpp"

namespace hrm
{

Shader::Shader(MTL::Device* device, const ShaderDesc& desc)
    : m_pDevice(device), m_Desc(desc)
{
    assert(device != nullptr);

    NS::String* pSource = NS::String::string(desc.sourceCode.c_str(), NS::StringEncoding::UTF8StringEncoding);
    assert(pSource != nullptr);

    NS::Error* pError = nullptr;
    m_pLibrary = device->newLibrary(pSource, nullptr, &pError);
    if (pError != nullptr)
    {
        LOG_ERROR("Failed to create library: %s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    m_pHandle = m_pLibrary->newFunction(NS::String::string(desc.entryPoint.c_str(), NS::StringEncoding::UTF8StringEncoding));
}

Shader::~Shader()
{
    if (m_pLibrary != nullptr)
    {
        m_pLibrary->release();
    }
    if (m_pHandle != nullptr)
    {
        m_pHandle->release();
    }
}

}  // namespace hrm

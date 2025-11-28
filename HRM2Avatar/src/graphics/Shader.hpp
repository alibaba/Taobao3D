#pragma once

#include <Metal/Metal.hpp>

namespace hrm
{

struct ShaderDesc
{
    std::string sourceCode;
    std::string entryPoint;
    std::string name;
};


class Shader
{
public:
    Shader(MTL::Device* device, const ShaderDesc& desc);
    ~Shader();

    MTL::Function* GetHandle() const { return m_pHandle; }
    const ShaderDesc& GetDesc() const { return m_Desc; }

private:
    MTL::Device*   m_pDevice { nullptr };
    MTL::Library*  m_pLibrary { nullptr };
    MTL::Function* m_pHandle { nullptr };

    ShaderDesc m_Desc;
};

}  // namespace hrm

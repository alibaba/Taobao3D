#include "GaussianAsset.hpp"

#include "Renderer.hpp"

namespace hrm
{

GaussianAsset::~GaussianAsset()
{
    if (m_ColorTexture != nullptr)
    {
        m_ColorTexture->release();
    }
}

void GaussianAsset::SetChunkData(const uint8_t* data, uint32_t size)
{
    m_ChunkData = m_pRenderer->CreateBuffer(size, "Chunk Data Buffer");
    m_ChunkData->SetData(data, size);
}

void GaussianAsset::SetPosData(const uint8_t* data, uint32_t size)
{
    m_PosData = m_pRenderer->CreateBuffer(size, "Pos Data Buffer");
    m_PosData->SetData(data, size);
}

void GaussianAsset::SetOtherData(const uint8_t* data, uint32_t size)
{
    m_OtherData = m_pRenderer->CreateBuffer(size, "Other Data Buffer");
    m_OtherData->SetData(data, size);
}

void GaussianAsset::SetColorData(const uint8_t* data, uint32_t size)
{
    Graphics* pGraphics = m_pRenderer->GetGraphics();
    m_ColorTexture = pGraphics->CreateSimpleTexture2D(m_ColorTextureWidth, m_ColorTextureHeight, MTL::PixelFormat::PixelFormatRGBA8Unorm);
    m_ColorTexture->setLabel(NS::String::string("Gaussian Color Texture", NS::StringEncoding::UTF8StringEncoding));
    pGraphics->CopyToDeviceTexture(data, size, m_ColorTexture);
}

void GaussianAsset::SetSHData(const uint8_t* data, uint32_t size)
{
    m_SHData = m_pRenderer->CreateBuffer(size, "SH Data Buffer");
    m_SHData->SetData(data, size);
}

void GaussianAsset::SetIdxData(const uint8_t* data, uint32_t size)
{
    m_IdxData = m_pRenderer->CreateBuffer(size, "Idx Data Buffer");
    m_IdxData->SetData(data, size);
}

void GaussianAsset::SetFacePropData(const uint8_t* data, uint32_t size)
{
    m_FacePropData = m_pRenderer->CreateBuffer(size, "Face Prop Data Buffer");
    m_FacePropData->SetData(data, size);
}

void GaussianAsset::SetGaussianPropData(const uint8_t* data, uint32_t size)
{
    m_GaussianPropData = m_pRenderer->CreateBuffer(size, "Gaussian Prop Data Buffer");
    m_GaussianPropData->SetData(data, size);
}

void GaussianAsset::SetColorTextureSize(uint32_t width, uint32_t height)
{
    m_ColorTextureWidth = width;
    m_ColorTextureHeight = height;
}

}  // namespace hrm

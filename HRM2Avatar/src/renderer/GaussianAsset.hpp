#pragma once

#include "Buffer.hpp"
#include "Metal/MTLTexture.hpp"

namespace hrm
{

class Renderer;

enum class VectorFormat : uint8_t
{
    Float32, // 12 bytes: 32F.32F.32F
    Norm16, // 6 bytes: 16.16.16
    Norm11, // 4 bytes: 11.10.11
    Norm6   // 2 bytes: 6.5.5
};

enum class SHFormat : uint8_t
{
    Float32,
    Float16,
    Norm11,
    Norm6,
    Cluster64k,
    Cluster32k,
    Cluster16k,
    Cluster8k,
    Cluster4k,
};

class GaussianAsset
{
public:
    GaussianAsset(Renderer* pRenderer) : m_pRenderer(pRenderer) {}
    ~GaussianAsset();

    void SetSplatCount(uint32_t splatCount) { m_SplatCount = splatCount; }
    uint32_t GetSplatCount() const { return m_SplatCount; }

    void SetPositionFormat(VectorFormat positionFormat) { m_PositionFormat = positionFormat; }
    VectorFormat GetPositionFormat() const { return m_PositionFormat; }

    void SetScaleFormat(VectorFormat scaleFormat) { m_ScaleFormat = scaleFormat; }
    VectorFormat GetScaleFormat() const { return m_ScaleFormat; }

    void SetSHFormat(SHFormat shFormat) { m_SHFormat = shFormat; }
    SHFormat GetSHFormat() const { return m_SHFormat; }

    void SetChunkData(const uint8_t* data, uint32_t size);
    Buffer* GetChunkData() const { return m_ChunkData.get(); }

    void SetPosData(const uint8_t* data, uint32_t size);
    Buffer* GetPosData() const { return m_PosData.get(); }

    void SetOtherData(const uint8_t* data, uint32_t size);
    Buffer* GetOtherData() const { return m_OtherData.get(); }

    void SetColorData(const uint8_t* data, uint32_t size);
    MTL::Texture* GetColorTexture() const { return m_ColorTexture; }

    void SetSHData(const uint8_t* data, uint32_t size);
    Buffer* GetSHData() const { return m_SHData.get(); }

    void SetIdxData(const uint8_t* data, uint32_t size);
    Buffer* GetIdxData() const { return m_IdxData.get(); }

    void SetFacePropData(const uint8_t* data, uint32_t size);
    Buffer* GetFacePropData() const { return m_FacePropData.get(); }
    
    void SetGaussianPropData(const uint8_t* data, uint32_t size);
    Buffer* GetGaussianPropData() const { return m_GaussianPropData.get(); }

    void SetColorTextureSize(uint32_t width, uint32_t height);

private:
    Renderer* m_pRenderer { nullptr };

    uint32_t m_SplatCount { 0u };
    VectorFormat m_PositionFormat { VectorFormat::Norm16 };
    VectorFormat m_ScaleFormat { VectorFormat::Norm16 };
    SHFormat m_SHFormat { SHFormat::Norm11 };
    
    std::unique_ptr<Buffer> m_ChunkData;
    std::unique_ptr<Buffer> m_PosData;
    std::unique_ptr<Buffer> m_OtherData;  // rotation and scale
    std::unique_ptr<Buffer> m_SHData;

    uint32_t m_ColorTextureWidth { 0u };
    uint32_t m_ColorTextureHeight { 0u };
    MTL::Texture* m_ColorTexture { nullptr };

    std::unique_ptr<Buffer> m_IdxData;  // splatIdx to triangle index
    std::unique_ptr<Buffer> m_FacePropData;  // triangle face flag data
    std::unique_ptr<Buffer> m_GaussianPropData;  // splat face flag data
};

}

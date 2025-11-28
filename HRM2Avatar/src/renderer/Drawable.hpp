#pragma once

#include <simd/simd.h>
#include "Metal/Metal.hpp"
#include "QuartzCore/CAMetalLayer.hpp"

namespace hrm
{

class Drawable
{
public:
    explicit Drawable(MTL::Device* device)
    : m_pDevice{ device }
    {
        
    }
    virtual ~Drawable() = default;
    
    virtual uint32_t GetViewCount() const = 0;
    virtual MTL::Texture* GetTexture(uint32_t viewIndex) const = 0;
    virtual MTL::Texture* GetDepthTexture(uint32_t viewIndex) const = 0;
    virtual MTL::RasterizationRateMap* GetRasterizationRateMap(uint32_t viewIndex) const { return nullptr; }
    virtual simd::float2 GetPhysicalSize() const = 0;
    virtual simd::float2 GetLogicalSize() const = 0;
private:
    MTL::Device* m_pDevice { nullptr };
};

// use for ios/osx
class MetalDrawable : public Drawable
{
public:
    MetalDrawable(MTL::Device* device, CA::MetalDrawable* handle)
    : Drawable(device), m_Handle(handle)
    {}
    
    uint32_t GetViewCount() const override { return 1u; }
    
    MTL::Texture* GetTexture(uint32_t viewIndex) const override { return viewIndex==0 ? m_Handle->texture() : nullptr; }
    MTL::Texture* GetDepthTexture(uint32_t viewIndex) const override { return nullptr; }
    CA::MetalDrawable* GetHandle() const { return m_Handle; }
    simd::float2 GetPhysicalSize() const override { return simd_make_float2(m_Handle->texture()->width(), m_Handle->texture()->height()); }
    simd::float2 GetLogicalSize() const override;
private:
    CA::MetalDrawable* m_Handle { nullptr };
};

#ifdef PLATFORM_VISIONOS
class MetalVisionDrawable : public Drawable
{
public:
    MetalVisionDrawable(MTL::Device* device, uint64_t handle);
    
    uint32_t GetViewCount() const override { return m_ViewCount; }
    MTL::Texture* GetTexture(uint32_t viewIndex) const override;
    MTL::Texture* GetDepthTexture(uint32_t viewIndex) const override;
    MTL::RasterizationRateMap* GetRasterizationRateMap(uint32_t viewIndex) const override;
    uint64_t GetHandle() const { return m_Handle; }
    simd::float2 GetPhysicalSize() const override { return simd_make_float2(GetTexture(0)->width(), GetTexture(0)->height()); }
    simd::float2 GetLogicalSize() const override;
    
private:
    uint64_t m_Handle { UINT64_MAX };
    uint32_t m_ViewCount { UINT32_MAX };
};
#endif

}  // namespace hrm

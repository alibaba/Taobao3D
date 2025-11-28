#include "Drawable.hpp"
#include "MetalHelper.hpp"


namespace hrm
{

simd::float2 MetalDrawable::GetLogicalSize() const
{
    MTL::RasterizationRateMap* rrm = GetRasterizationRateMap(0);
    if (rrm == nullptr)
        return GetPhysicalSize();

    auto screenSize = rrm->screenSize();
    return simd_make_float2(screenSize.width, screenSize.height);
}

#ifdef PLATFORM_VISIONOS
MetalVisionDrawable::MetalVisionDrawable(MTL::Device* device, uint64_t handle)
    : Drawable(device), m_Handle(handle)
{
    m_ViewCount = MetalHelper::GetDrawableViewCount(m_Handle);
}

MTL::Texture* MetalVisionDrawable::GetTexture(uint32_t viewIndex) const
{
    return MetalHelper::GetMetalColorTexture(m_Handle, viewIndex);
}

MTL::Texture* MetalVisionDrawable::GetDepthTexture(uint32_t viewIndex) const
{
//    return nullptr;
     return MetalHelper::GetMetalDepthTexture(m_Handle, viewIndex);
}

MTL::RasterizationRateMap* MetalVisionDrawable::GetRasterizationRateMap(uint32_t viewIndex) const
{
    //TODO: check if flip or not
    return MetalHelper::GetMetalRasterizationRateMap(m_Handle, viewIndex, false);
}

simd::float2 MetalVisionDrawable::GetLogicalSize() const
{
    MTL::RasterizationRateMap* rrm = GetRasterizationRateMap(0);
    if (rrm == nullptr)
        return GetPhysicalSize();

    auto screenSize = rrm->screenSize();
    return simd_make_float2(screenSize.width, screenSize.height);
}
#endif

}  // namespace hrm

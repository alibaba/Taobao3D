#pragma once

#include <cstdint>
#include <string>

#include <Metal/Metal.hpp>

namespace CA
{
class MetalDrawable;
}

namespace hrm
{

class MetalHelper
{
public:

    static uint32_t GetBytesPerPixel(MTL::PixelFormat pixelFormat);

    static CA::MetalDrawable* GetMetalDrawable(void* drawableHandle);
#ifdef PLATFORM_VISIONOS
    static uint32_t GetDrawableViewCount(uint64_t drawableHandle);
    static MTL::Texture* GetMetalColorTexture(uint64_t drawableHandle, uint32_t index);
    static MTL::Texture* GetMetalDepthTexture(uint64_t drawableHandle, uint32_t index);
    static MTL::RasterizationRateMap* GetMetalRasterizationRateMap(uint64_t drawableHandle, uint32_t index, bool flip);
    static void PresentWithDrawable(uint64_t drawableHandle, MTL::CommandBuffer* cbHandle);
#endif
};

// Returns bundle resources base path appended with "/assets/"
std::string GetBundleAssetsPath();

}  // namespace hrm

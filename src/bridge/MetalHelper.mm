#include "MetalHelper.hpp"

#ifdef PLATFORM_VISIONOS
#import <CompositorServices/CompositorServices.h>
#endif
#import <QuartzCore/CAMetalLayer.hpp>
#import <Foundation/Foundation.h>

// PROJECT_ROOT_DIR is defined by CMake
#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR ""
#endif

namespace hrm
{

uint32_t MetalHelper::GetBytesPerPixel(MTL::PixelFormat pixelFormat)
{
    switch (pixelFormat)
    {
    case MTL::PixelFormat::PixelFormatRGBA8Unorm:
    case MTL::PixelFormat::PixelFormatRGBA8Unorm_sRGB:
    case MTL::PixelFormat::PixelFormatRGBA8Snorm:
    case MTL::PixelFormat::PixelFormatRGBA8Uint:
    case MTL::PixelFormat::PixelFormatRGBA8Sint:
        return 4;
    case MTL::PixelFormat::PixelFormatRGBA16Float:
    case MTL::PixelFormat::PixelFormatRGBA16Snorm:
    case MTL::PixelFormat::PixelFormatRGBA16Unorm:
    case MTL::PixelFormat::PixelFormatRGBA16Uint:
    case MTL::PixelFormat::PixelFormatRGBA16Sint:
        return 8;
    case MTL::PixelFormat::PixelFormatRGBA32Float:
    case MTL::PixelFormat::PixelFormatRGBA32Sint:
    case MTL::PixelFormat::PixelFormatRGBA32Uint:
        return 16;
    default:
        assert(false);
        return 0;
    }
}

CA::MetalDrawable* MetalHelper::GetMetalDrawable(void* drawableHandle)
{
    return (__bridge CA::MetalDrawable*)((__bridge id)drawableHandle);
}

#ifdef PLATFORM_VISIONOS
uint32_t MetalHelper::GetDrawableViewCount(uint64_t drawableHandle)
{
    cp_drawable_t drawable_handle = (cp_drawable_t)(drawableHandle);
    size_t viewCount = cp_drawable_get_view_count(drawable_handle);
    return (uint32_t)viewCount;
}

MTL::Texture* MetalHelper::GetMetalColorTexture(uint64_t drawableHandle, uint32_t index)
{
    return (__bridge MTL::Texture*)(cp_drawable_get_color_texture((cp_drawable_t)drawableHandle, index));
}

MTL::Texture* MetalHelper::GetMetalDepthTexture(uint64_t drawableHandle, uint32_t index)
{
    return (__bridge MTL::Texture*)(cp_drawable_get_depth_texture((cp_drawable_t)drawableHandle, index));
}

MTL::RasterizationRateMap* MetalHelper::GetMetalRasterizationRateMap(uint64_t drawableHandle, uint32_t index, bool flip)
{
    auto map = flip ?
        cp_drawable_get_flipped_rasterization_rate_map((cp_drawable_t)drawableHandle, index) :
        cp_drawable_get_rasterization_rate_map((cp_drawable_t)drawableHandle, index);
    return (__bridge MTL::RasterizationRateMap*)(map);
}

void MetalHelper::PresentWithDrawable(uint64_t drawableHandle, MTL::CommandBuffer* cbHandle)
{
    id<MTLCommandBuffer> idCb = (__bridge id<MTLCommandBuffer>)(cbHandle);
    cp_drawable_encode_present((cp_drawable_t)drawableHandle, idCb);
}
#endif

std::string GetBundleAssetsPath()
{
#ifdef PLATFORM_OSX
    return std::string(PROJECT_ROOT_DIR) + "/assets/";
#else
    NSString* resPath = [NSBundle mainBundle].resourcePath;
    NSString* dir = [resPath stringByAppendingPathComponent:@"assets"]; // no trailing slash
    std::string path(dir.UTF8String);
    if (path.empty() || path.back() != '/')
    {
        path.push_back('/');
    }
    return path;
#endif
}

}  // namespace hrm


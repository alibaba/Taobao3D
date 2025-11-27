#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>

#if TARGET_OS_VISION
#import <CompositorServices/CompositorServices.h>
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RendererBridge : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)setDrawable:(void*)drawableHandle;
- (void)drawFrame;
- (void)logicalUpdate;

#if TARGET_OS_VISION
- (void)setVisionDrawable:(cp_drawable_t)drawableHandle;
- (void)setCameraMatrix:(uint32_t)eyeIdx viewMatrix:(simd_float4x4)viewMat projMatrix:(simd_float4x4)projMat;
#endif
@end

NS_ASSUME_NONNULL_END




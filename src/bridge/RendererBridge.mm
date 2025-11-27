#import "RendererBridge.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
//#import "Metal/Metal.hpp"
//#import "QuartzCore/CAMetalLayer.hpp"

#import "Renderer.hpp"

@interface RendererBridge () {
    hrm::Renderer* cppRenderer;
//    id<MTLDevice> objcDevice;
}
@end

@implementation RendererBridge

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        cppRenderer = new hrm::Renderer((__bridge MTL::Device *)device);
//        objcDevice = MTLCreateSystemDefaultDevice();
//        layer.device = objcDevice;
//        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
//
//        MTL::Device *device = (__bridge MTL::Device *)objcDevice;
//        CA::MetalLayer *metalLayer = (__bridge CA::MetalLayer *)layer;
//        cppRenderer = new Renderer(device, metalLayer);
    }
    return self;
}

- (void)setDrawable:(void *)drawableHandle{
    cppRenderer->SetDrawable(drawableHandle);
}

- (void)drawFrame {
    if (!cppRenderer) return;
    
    cppRenderer->Draw();
}

- (void)logicalUpdate {
    if (!cppRenderer) return;
    
    cppRenderer->LogicalUpdate();
}

#if TARGET_OS_VISION

/*
- (void)visionDrawFrame:(cp_layer_renderer_t)layerRenderer{
    if (!cppRenderer) return;
    // Get the next frame.
    cp_frame_t frame = cp_layer_renderer_query_next_frame(layerRenderer);
    if (frame == nullptr) { return; }
    
    // Fetch the predicted timing information.
    cp_frame_timing_t timing = cp_frame_predict_timing(frame);
    if (timing == nullptr) { return; }
    
    cp_frame_start_update(frame);
    // TODO: engine tick except for rendering
    cp_frame_end_update(frame);
    
    cp_time_wait_until(cp_frame_timing_get_optimal_input_time(timing));
    
    cp_drawable_t drawable = cp_frame_query_drawable(frame);
    if (drawable == nullptr) {
        return;
    }
    // TODO: complete
//    cp_frame_timing_t actualTiming = cp_drawable_get_frame_timing(drawable);
//    ar_device_anchor_t device_anchor = createPoseForTiming(actualTiming);
//    cp_drawable_set_device_anchor(drawable, device_anchor);
    cppRenderer->SetDrawable(drawable);
    
    cp_frame_start_submission(frame);
    // TODO: engine Rendering tick
    cppRenderer->Draw();
    cp_frame_end_submission(frame);
    
}
*/

- (void)setVisionDrawable:(cp_drawable_t)drawableHandle{
    if (!cppRenderer) return;
    cppRenderer->SetDrawable(drawableHandle);
}

- (void)setCameraMatrix:(uint32_t)eyeIdx viewMatrix:(simd_float4x4)viewMat projMatrix:(simd_float4x4)projMat{
    if (!cppRenderer) return;
    
    cppRenderer->SetCameraMatrix(eyeIdx, viewMat, projMat);
}
#endif

- (void)dealloc {
    delete cppRenderer;
    [super dealloc];
}

//- (void)resize:(CGSize)size {
//    if (!cppRenderer) return;
//    cppRenderer->resize((uint32_t)size.width, (uint32_t)size.height);
//}

@end




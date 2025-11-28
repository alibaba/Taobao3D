import SwiftUI
import MetalKit

final class _IOSMTKDelegate: NSObject, MTKViewDelegate {
    private let bridge: RendererBridge
    init(view: MTKView) {
        self.bridge = RendererBridge(device: view.device!)
        super.init()
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    func draw(in view: MTKView) {
        let layer = view.layer as! CAMetalLayer
        if let srgb = CGColorSpace(name: CGColorSpace.sRGB) {
            layer.colorspace = srgb
        }
        let drawable = layer.nextDrawable()
        if drawable == nil {
            return;
        }
        bridge.setDrawable(Unmanaged.passUnretained(drawable!).toOpaque());
        bridge.logicalUpdate()
        bridge.drawFrame()
    }
}

struct MetalView: UIViewRepresentable {    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.drawableSize = CGSize(width: 945, height: 2048)
        view.autoResizeDrawable = false
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 120
        view.framebufferOnly = false
        view.colorPixelFormat = .rgba16Float
        let delegate = _IOSMTKDelegate(view: view)
        context.coordinator.delegate = delegate
        view.delegate = delegate
        return view
    }
    func updateUIView(_ view: MTKView, context: Context) { }
    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var delegate: _IOSMTKDelegate? }
}

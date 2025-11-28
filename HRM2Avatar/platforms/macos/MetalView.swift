import SwiftUI
import MetalKit

final class _MacMTKDelegate: NSObject, MTKViewDelegate {
    private let bridge: RendererBridge
    init(view: MTKView) {
        self.bridge = RendererBridge(device: view.device!)
        super.init()
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    func draw(in view: MTKView) {
        let layer = view.layer as! CAMetalLayer
        if layer.contentsScale != 1.0 {
            layer.contentsScale = 1.0
            view.drawableSize = view.bounds.size
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

struct MetalView: NSViewRepresentable {
    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 120
        view.framebufferOnly = false
        if let srgb = CGColorSpace(name: CGColorSpace.sRGB) {
            view.colorspace = srgb
        }

//        view.colorPixelFormat = .rgba8Unorm_srgb
        view.colorPixelFormat = .rgba16Float
        let delegate = _MacMTKDelegate(view: view)
        context.coordinator.delegate = delegate
        view.delegate = delegate
        return view
    }
    func updateNSView(_ view: MTKView, context: Context) { }
    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator {
        var delegate: _MacMTKDelegate?
    }
}

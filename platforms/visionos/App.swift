import SwiftUI
import CompositorServices
import ARKit

struct ContentStageConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        configuration.depthFormat = .depth32Float
        configuration.colorFormat = .rgba16Float
//        configuration.colorUsage = [.renderTarget, .shaderRead, .shaderWrite]

        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled

        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)

        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
    }
}

@available(visionOS 2.0, *)
final class RendererTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
          job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        return UnownedTaskExecutor(ordinary: self)
    }

    static var shared: RendererTaskExecutor = RendererTaskExecutor()
}

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

@available(visionOS 2.0, *)
actor VisionRenderer {
    
    let layerRenderer: LayerRenderer
    let appModel: AppModel
    let bridge: RendererBridge
    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    
    init(_ layerRenderer: LayerRenderer, appModel: AppModel) {
        self.layerRenderer = layerRenderer
        self.appModel = appModel
        bridge = RendererBridge(device: layerRenderer.device)
        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
    }
    
    @MainActor
    static func startRenderLoop(_ layerRenderer: LayerRenderer, appModel: AppModel) {
        Task(executorPreference: RendererTaskExecutor.shared) {
            let renderer = VisionRenderer(layerRenderer, appModel: appModel)
            await renderer.startARSession()
            await renderer.renderLoop()
        }
    }
    
    private func startARSession() async {
        do {
            try await arSession.run([worldTracking])
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }
    
    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                Task { @MainActor in
                    appModel.immersiveSpaceState = .closed
                }
                return
            } else if layerRenderer.state == .paused {
                Task { @MainActor in
                    appModel.immersiveSpaceState = .inTransition
                }
                layerRenderer.waitUntilRunning()
                continue
            } else {
                Task { @MainActor in
                    if appModel.immersiveSpaceState != .open {
                        appModel.immersiveSpaceState = .open
                    }
                }
                autoreleasepool {
                    /// Per frame updates hare
                    guard let frame = layerRenderer.queryNextFrame() else { return }
                    frame.startUpdate()

                    bridge.logicalUpdate()

                    frame.endUpdate()
                    
                    guard let timing = frame.predictTiming() else { return }
                    LayerRenderer.Clock().wait(until: timing.optimalInputTime)
                    guard let drawable = frame.queryDrawable() else { return }
                    drawable.depthRange = .init(5.0, 0.1)
                    
                    let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
                    let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)
                    if deviceAnchor == nil {
                        print("error: can not query device anchor!, worldTracking's state: %@", worldTracking.state)
                    }
                    drawable.deviceAnchor = deviceAnchor
                    bridge.setVisionDrawable(drawable)
                    
                    let headToWorld = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4
                    
                    for viewIdx in 0...drawable.views.count-1 {
                        let eyeToHead = drawable.views[viewIdx].transform
                        bridge.setCameraMatrix(UInt32(viewIdx), viewMatrix: (headToWorld * eyeToHead).inverse, projMatrix: drawable.computeProjection(viewIndex: viewIdx))
                    }
                    
                    frame.startSubmission()
                    bridge.drawFrame()
                    frame.endSubmission()

                }
            }
        }
    }
    
}





@available(visionOS 2.0, *)
@main
struct SwiftVisionTestApp: App {

    @State private var appModel = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appModel)
        }
        .windowStyle(.volumetric)

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            CompositorLayer(configuration: ContentStageConfiguration()) { @MainActor layerRenderer in
                VisionRenderer.startRenderLoop(layerRenderer, appModel: appModel)
            }
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed, .full)
    }
}

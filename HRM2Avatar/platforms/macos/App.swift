import SwiftUI

@main
struct MacApp: App {
    var body: some Scene {
        WindowGroup {
            MetalView()
                .frame(width: 1200, height: 796)
        }
    }
}



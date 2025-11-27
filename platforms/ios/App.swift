import SwiftUI

@available(iOS 14.0, *)
@main
struct MacApp: App {
    var body: some Scene {
        WindowGroup {
            MetalView()
                .edgesIgnoringSafeArea(.all)
        }
    }
}



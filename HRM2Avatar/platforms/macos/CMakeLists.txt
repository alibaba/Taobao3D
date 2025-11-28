if(DEFINED TARGET_PLATFORM AND "${TARGET_PLATFORM}" STREQUAL "PLATFORM_OSX")
    add_executable(avatar-osx MACOSX_BUNDLE
        App.swift
        MetalView.swift
    )

    # Swift bridging
    set_target_properties(avatar-osx PROPERTIES
        XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_SOURCE_DIR}/src/bridge/Bridging-Header.h"
    )

    # Link dependencies
    target_link_libraries(avatar-osx PRIVATE AVATAR_CORE)
endif()


if(DEFINED TARGET_PLATFORM AND "${TARGET_PLATFORM}" STREQUAL "PLATFORM_VISIONOS")
    add_executable(avatar-visionos MACOSX_BUNDLE
        App.swift
        AppModel.swift
        ContentView.swift
        )

    set_target_properties(avatar-visionos PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/visionOS-Info.plist"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.example.avatar"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.example.avatar"
    XCODE_ATTRIBUTE_PRODUCT_NAME "HRM2Avatar"
    XCODE_ATTRIBUTE_MARKETING_VERSION "1.0.0"
    XCODE_ATTRIBUTE_CURRENT_PROJECT_VERSION "1"
    # XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "YOURTEAMID"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"

    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_SOURCE_DIR}/src/bridge/Bridging-Header.h"
    )

    target_link_libraries(avatar-visionos PRIVATE
        AVATAR_CORE
        "-framework CompositorServices"
        -lc++)

    # Add assets as bundle resources and preserve subdirectory structure
    file(GLOB_RECURSE APP_ASSETS "${CMAKE_SOURCE_DIR}/assets/*")
    if(APP_ASSETS)
        foreach(ASSET_FILE ${APP_ASSETS})
            file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}/assets" "${ASSET_FILE}")
            get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
            set(DEST_DIR "Resources/assets")
            if(NOT REL_DIR STREQUAL "")
                set(DEST_DIR "Resources/assets/${REL_DIR}")
            endif()
            message(STATUS "Adding asset: ${ASSET_FILE} to ${DEST_DIR}")
            set_source_files_properties("${ASSET_FILE}" PROPERTIES MACOSX_PACKAGE_LOCATION "${DEST_DIR}")
        endforeach()
        target_sources(avatar-visionos PRIVATE ${APP_ASSETS})
    endif()
endif()
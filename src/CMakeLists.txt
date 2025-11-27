file(GLOB_RECURSE AVATAR_CORE_CPP
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
)

file(GLOB_RECURSE AVATAR_CORE_MM
    "${CMAKE_SOURCE_DIR}/src/*.mm"
    "${CMAKE_SOURCE_DIR}/src/*.h"
)

add_library(AVATAR_CORE STATIC
    ${AVATAR_CORE_CPP}
    ${AVATAR_CORE_MM}
)

set_source_files_properties(
    ${AVATAR_CORE_MM}
    PROPERTIES
    COMPILE_FLAGS "-fno-objc-arc"
)

target_link_libraries(AVATAR_CORE PUBLIC THIRD_LIB)

if(TARGET_PLATFORM STREQUAL "PLATFORM_OSX")
    set(MNN_PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-MacOS-CPU-GPU/Static" CACHE PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-MacOS-CPU-GPU/Static")
    find_library(MNNFramework NAMES MNN PATHS ${MNN_PATH})
    target_link_libraries(AVATAR_CORE PRIVATE ${MNNFramework}  
                                "-framework Metal"
                                "-framework Foundation")
elseif(TARGET_PLATFORM STREQUAL "PLATFORM_IOS")
    set(MNN_PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-iOS-CPU-GPU/Static" CACHE PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-iOS-CPU-GPU/Static")
    find_library(MNNFramework NAMES MNN PATHS ${MNN_PATH})
    target_link_libraries(AVATAR_CORE PRIVATE ${MNNFramework}  
                            "-framework Metal"
                            "-framework Foundation")
elseif(TARGET_PLATFORM STREQUAL "PLATFORM_VISIONOS")
    set(MNN_PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-visionOS-CPU-GPU/Static" CACHE PATH "${CMAKE_SOURCE_DIR}/third/mnn/MNN-visionOS-CPU-GPU/Static")
    find_library(MNNFramework NAMES MNN PATHS ${MNN_PATH})
    target_link_libraries(AVATAR_CORE PRIVATE ${MNNFramework}  
                            "-framework Metal"
                            "-framework Foundation")
endif()


cmake_minimum_required(VERSION 3.29)

project(HRM2Avatar LANGUAGES Swift C CXX OBJC OBJCXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/third/metal-cpp
    ${CMAKE_SOURCE_DIR}/third/tiny-object-loader
    ${CMAKE_SOURCE_DIR}/third/nlohmann-json
    
    ${CMAKE_SOURCE_DIR}/src/renderer
    ${CMAKE_SOURCE_DIR}/src/graphics
    ${CMAKE_SOURCE_DIR}/src/animation
    ${CMAKE_SOURCE_DIR}/src/bridge
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/neural_network
)

add_compile_definitions(${TARGET_PLATFORM})

# Define project root path as a C/C++ string macro
add_compile_definitions(PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

add_subdirectory(third)
add_subdirectory(src)
add_subdirectory(platforms)
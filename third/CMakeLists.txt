include(FetchContent)

# build metal-cpp lib

# Library definition
add_library(METAL_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-definition.cpp
        )

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        )
set_target_properties(METAL_CPP PROPERTIES FOLDER "third")

# build nlohmann-json
add_library(NLOHMANN_JSON
        ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann-json/json.cpp
)
set_target_properties(NLOHMANN_JSON PROPERTIES FOLDER "third")

# build MNN via FetchContent
set(MNN_FETCH_VERSION "3.2.5")
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TRAIN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
set(MNN_EVALUATION OFF CACHE BOOL "" FORCE)
set(MNN_METAL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    MNN
    GIT_REPOSITORY https://github.com/alibaba/MNN.git
    GIT_TAG ${MNN_FETCH_VERSION}
    GIT_PROGRESS TRUE
)
FetchContent_GetProperties(MNN)
if(NOT mnn_POPULATED)
    message(STATUS "[THIRD] Fetching MNN ${MNN_FETCH_VERSION}...")
endif()
FetchContent_MakeAvailable(MNN)

# set MNN include directories
if(DEFINED mnn_SOURCE_DIR)
    target_include_directories(MNN INTERFACE ${mnn_SOURCE_DIR}/include)
endif()

# move MNN targets to third/mnn folder
set(MNN_TARGET_FOLDER "third/mnn")
set(_MNN_ALL_TARGETS "")

if(DEFINED mnn_SOURCE_DIR AND EXISTS "${mnn_SOURCE_DIR}")
    get_property(_MNN_SOURCE_TARGETS DIRECTORY ${mnn_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND _MNN_ALL_TARGETS ${_MNN_SOURCE_TARGETS})
endif()
if(DEFINED mnn_BINARY_DIR AND EXISTS "${mnn_BINARY_DIR}")
    get_property(_MNN_BINARY_TARGETS DIRECTORY ${mnn_BINARY_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND _MNN_ALL_TARGETS ${_MNN_BINARY_TARGETS})
endif()

if(TARGET MNN_Express)
        list(APPEND _MNN_ALL_TARGETS MNN_Express)
endif()

list(REMOVE_DUPLICATES _MNN_ALL_TARGETS)
foreach(MNN_SUBTARGET IN LISTS _MNN_ALL_TARGETS)
    if(TARGET ${MNN_SUBTARGET})
        set_target_properties(${MNN_SUBTARGET} PROPERTIES FOLDER "${MNN_TARGET_FOLDER}")
    endif()
endforeach()

# build third-party umbrella static library
add_library(THIRD_LIB INTERFACE)

target_link_libraries(THIRD_LIB INTERFACE METAL_CPP NLOHMANN_JSON MNN)
if(TARGET MNN_Express)
    target_link_libraries(THIRD_LIB INTERFACE MNN_Express)
endif()

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


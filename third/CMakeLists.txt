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

# build third-party umbrella static library
add_library(THIRD_LIB INTERFACE)

target_link_libraries(THIRD_LIB INTERFACE METAL_CPP NLOHMANN_JSON)

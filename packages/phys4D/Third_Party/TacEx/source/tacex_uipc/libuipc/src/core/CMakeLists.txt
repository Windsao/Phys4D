find_package(Eigen3 REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt CONFIG REQUIRED)

find_package(cppitertools CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_path(DYLIB_INCLUDE_DIRS "dylib.hpp")
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
find_package(cpptrace CONFIG REQUIRED)

add_library(uipc_core SHARED)
add_library(uipc::core ALIAS uipc_core)
target_compile_features(uipc_core PUBLIC cxx_std_20)




target_compile_definitions(uipc_core PRIVATE UIPC_CORE_EXPORT_DLL=1)

target_compile_definitions(uipc_core PUBLIC
    $<$<CONFIG:Debug>:UIPC_RUNTIME_CHECK=1>
    $<$<CONFIG:RelWithDebInfo>:UIPC_RUNTIME_CHECK=1>
    $<$<CONFIG:Release>:UIPC_RUNTIME_CHECK=1>
)
target_compile_definitions(uipc_core PUBLIC "-DUIPC_PROJECT_DIR=R\"(${PROJECT_SOURCE_DIR})\"")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_MAJOR=${UIPC_VERSION_MAJOR}")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_MINOR=${UIPC_VERSION_MINOR}")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_PATCH=${UIPC_VERSION_PATCH}")




uipc_target_add_include_files(uipc_core)
target_include_directories(uipc_core PRIVATE "${DYLIB_INCLUDE_DIRS}")
target_include_directories(uipc_core PRIVATE "${TINYGLTF_INCLUDE_DIRS}")

file(GLOB SOURCE "*.h" "*.cpp")
target_sources(uipc_core PRIVATE ${SOURCE})





target_link_libraries(uipc_core PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    fmt::fmt
    cppitertools::cppitertools
    magic_enum::magic_enum
    cpptrace::cpptrace
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    target_link_libraries(uipc_core PRIVATE dl)
endif()

if(MSVC)

    target_compile_options(uipc_core PRIVATE "/bigobj")

    target_compile_definitions(uipc_core PUBLIC NOMINMAX)
endif()




add_subdirectory(common)
add_subdirectory(backend)
add_subdirectory(geometry)
add_subdirectory(constitution)
add_subdirectory(builtin)
add_subdirectory(core)
add_subdirectory(diff_sim)




uipc_target_set_output_directory(uipc_core)
uipc_target_set_rpath(uipc_core)





file(GLOB_RECURSE SOURCE_GROUP_FILES "*.h" "*.cpp" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})












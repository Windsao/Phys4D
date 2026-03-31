if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(OpenVDB REQUIRED)



add_library(uipc_vdb SHARED)
add_library(uipc::vdb ALIAS uipc_vdb)
target_compile_features(uipc_vdb PUBLIC cxx_std_20)
target_compile_definitions(uipc_vdb INTERFACE UIPC_WITH_VDB_SUPPORT=1)
target_compile_definitions(uipc_vdb PRIVATE UIPC_VDB_EXPORT_DLL=1)
target_include_directories(uipc_vdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(uipc_vdb PRIVATE
    uipc::core
    OpenVDB::openvdb
)

if(APPLE)

    find_package(blosc CONFIG REQUIRED)
    target_link_libraries(uipc_vdb PRIVATE blosc_static)

    target_link_options(uipc_vdb PRIVATE "LINKER:-no_warn_duplicate_libraries")
endif()

uipc_target_set_output_directory(uipc_vdb)
uipc_target_set_rpath(uipc_vdb)

file(GLOB_RECURSE SOURCES "*.h" "*.hpp" "*.cpp" "*.inl")

target_sources(uipc_vdb PRIVATE ${SOURCES})

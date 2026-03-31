find_package(pxr REQUIRED CONFIG PATHS ${UIPC_USD_INSTALL_DIR} NO_DEFAULT_PATH)

uipc_info("Found Pxr Version ${PXR_VERSION} from [${UIPC_USD_INSTALL_DIR}]")

add_library(uipc_usd SHARED)
add_library(uipc::usd ALIAS uipc_usd)

target_include_directories(uipc_usd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

uipc_target_add_include_files(uipc_usd)





target_link_libraries(uipc_usd PRIVATE
    usd
    usdGeom
    usdPhysics
)


target_link_libraries(uipc_usd PUBLIC
    uipc::core
    uipc::geometry)

target_compile_definitions(uipc_usd PRIVATE UIPC_USD_EXPORT_DLL=1)

file(GLOB_RECURSE SOURCES "*.cpp" "*.h" "details/*.inl")
target_sources(uipc_usd PRIVATE ${SOURCES})

uipc_target_set_output_directory(uipc_usd)
uipc_target_set_rpath(uipc_usd)




file(GLOB_RECURSE SOURCE_GROUP_FILES "*.h" "*.cpp" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})

find_package(bgfx CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_library(uipc_gui STATIC)
add_library(uipc::gui ALIAS uipc_gui)




file(GLOB SOURCE "*.cpp" "*.h" "details/*.inl")
target_sources(uipc_gui PRIVATE ${SOURCE})




target_link_libraries(uipc_gui PRIVATE
    imgui::imgui
    bgfx::bx
    bgfx::bgfx
    bgfx::bimg
    bgfx::bimg_decode
    glfw
)

target_link_libraries(uipc_gui PUBLIC uipc::core)
include(uipc_utils)

uipc_target_add_backend_dependency(uipc_gui)
uipc_target_add_include_files(uipc_gui)
uipc_target_set_output_directory(uipc_gui)
uipc_target_set_rpath(uipc_gui)





add_subdirectory(common)





file(GLOB_RECURSE ALL_SOURCE "*.cpp" "*.h" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${ALL_SOURCE})

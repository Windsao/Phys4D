
uipc_add_backend(none)

target_include_directories(none PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SOURCES "*.cpp" "*.h" "details/*.inl")

target_sources(none PRIVATE ${SOURCES})


source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCES})

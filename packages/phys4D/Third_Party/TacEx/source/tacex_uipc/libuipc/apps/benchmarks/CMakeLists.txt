find_package(Catch2 CONFIG REQUIRED)



function(uipc_add_benchmark name)
    add_executable(${name} ${ARGN})
    target_sources(${name} PRIVATE ${benchmark_main_cpp})
    target_link_libraries(${name} PRIVATE
        uipc::uipc
        app
        Catch2::Catch2WithMain)

    set_property(TARGET ${name} PROPERTY FOLDER "apps/benchmarks")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "uipc_benchmark_${name}")


    uipc_target_add_backend_dependency(${name})

    uipc_target_set_output_directory(${name})
endfunction()

add_subdirectory(basic)
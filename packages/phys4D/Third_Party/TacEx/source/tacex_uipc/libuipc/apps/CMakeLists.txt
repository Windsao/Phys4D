
add_subdirectory(app)


if(UIPC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(UIPC_BUILD_TESTS)
    add_subdirectory(tests)
endif()


if(UIPC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
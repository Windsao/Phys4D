cmake_minimum_required(VERSION 3.26)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/uipc_utils.cmake)




set(UIPC_VERSION_MAJOR 0)
set(UIPC_VERSION_MINOR 9)
set(UIPC_VERSION_PATCH 0)




option(UIPC_USING_LOCAL_VCPKG "Using local vcpkg" ON)
option(UIPC_BUILD_GUI "Build GUI, turn it off if you're server only" OFF)
option(UIPC_BUILD_PYBIND "Build pyuipc" OFF)
option(UIPC_BUILD_EXAMPLES "Build examples" ON)
option(UIPC_BUILD_TESTS "Build tests" ON)
option(UIPC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(UIPC_DEV_MODE "Enable developer mode" OFF)
option(UIPC_WITH_USD_SUPPORT "Build with Universal Scene Description (USD) support" OFF)
option(UIPC_WITH_VDB_SUPPORT "Build with Volumetric DataBase (VDB) support" OFF)

set(UIPC_PYTHON_EXECUTABLE_PATH "" CACHE PATH "Python executable path")
set(UIPC_USD_INSTALL_DIR "" CACHE PATH
        "USD installation directory: the folder that contains pxrConfig.cmake, e.g. ~/Software/Usd")




if(APPLE)
    set(CUDA_BACKEND_ON OFF)
else()
    set(CUDA_BACKEND_ON ON)
endif()
option(UIPC_WITH_CUDA_BACKEND "Build with CUDA backend" ${CUDA_BACKEND_ON})





uipc_show_logo()
uipc_find_python_executable_path()
uipc_show_options()
uipc_config_vcpkg_install()








project(uipc)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_subdirectory(external)


add_subdirectory(src)


add_subdirectory(apps)

cmake_minimum_required(VERSION 3.18)

include(CMakeDependentOption)
include(FetchContent)

project(pybind_gmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Setting Release as the type of build." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT LIB_FOLDER)
    set(LIB_FOLDER ${CMAKE_SOURCE_DIR}/lib)
endif(NOT LIB_FOLDER)

CMAKE_DEPENDENT_OPTION(CREATE_BINDINGS "If `ON`: Creates pybind11 bindings for the core GMM algorithm. Ingored if cross compiling." ON "NOT CMAKE_CROSSCOMPILING" OFF)
option(CREATE_MAIN "If `ON`: Creates a testable main file." ON)
option(CREATE_TESTS "If `ON`: Creates the C++ tests using googletest." ON)
option(DEBUG_STATEMENTS "If `ON`: Prints information about the algorithm to the command line." OFF)
option(INTENSE_DEBUG_STATEMENTS "If `ON`: Prints even more information about the algorithm to the command line. Slows the algorithm down immensly." OFF)

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)

if(MSVC)
    add_compile_options(
        /std:c++17
        $<$<CONFIG:RELWITHDEBINFO>:/O2>
        $<$<CONFIG:DEBUG>:/Od>
    )
else()
    add_compile_options(
        -std=c++17
        $<$<CONFIG:RELWITHDEBINFO>:-Ofast>
        $<$<CONFIG:RELWITHDEBINFO>:-march=native>
        $<$<CONFIG:DEBUG>:-O0>
        $<$<CONFIG:DEBUG>:-ggdb3>
    )
endif(MSVC)

include_directories(include)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
  SOURCE_DIR        "${LIB_FOLDER}/eigen"
  GIT_TAG           3.4.0
)
FetchContent_Declare(
  LBFGSpp
  GIT_REPOSITORY    https://github.com/yixuan/LBFGSpp.git
  SOURCE_DIR        "${LIB_FOLDER}/LBFGSpp"
  GIT_TAG           v0.3.0
)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY    https://github.com/gabime/spdlog.git
  SOURCE_DIR        "${LIB_FOLDER}/spdlog"
  GIT_TAG           v1.13.0
)
FetchContent_MakeAvailable(eigen LBFGSpp spdlog)
include_directories(lib/eigen/include)
include_directories(lib/LBFGSpp/include)
include_directories(lib/spdlog/include)

if(CREATE_BINDINGS)
    set(PYBIND11_PYTHON_VERSION 3.10)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        SOURCE_DIR "${LIB_FOLDER}/pybind11"
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    include_directories(lib/pybind11/include)

    pybind11_add_module(mc3d_trecsim_gmm src/bindings.cpp)

    target_compile_definitions(mc3d_trecsim_gmm PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    target_link_libraries(mc3d_trecsim_gmm PRIVATE Eigen3::Eigen)
    list(APPEND PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}")

    add_custom_command(TARGET mc3d_trecsim_gmm POST_BUILD
        VERBATIM COMMAND python -c "import os; import sys; sys.path.insert(1, os.getcwd()); from pybind11_stubgen import main; import mc3d_trecsim_gmm; sys.argv[0] = 'pybind11-stubgen'; sys.argv.append('--ignore-all-errors'); sys.argv.append('-o'); sys.argv.append('.'); sys.argv.append('mc3d_trecsim_gmm'); main()" || (exit 0)
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Creating stubs for MC3D-TRECSIM"
    )
endif(CREATE_BINDINGS)

if(CREATE_MAIN)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_executable(main src/main.cpp)
    target_link_libraries(main PRIVATE Eigen3::Eigen)
endif(CREATE_MAIN)

if(CREATE_TESTS)
    add_subdirectory(test/cpp)
endif(CREATE_TESTS)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app LANGUAGES CXX VERSION 1.0)

set(CMAKE_PREFIX_PATH "../gmm/lib/libtorch/")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 20)
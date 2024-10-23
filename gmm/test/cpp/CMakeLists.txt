include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  SOURCE_DIR        "${LIB_FOLDER}/googletest"
  GIT_TAG           v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test GTest::gtest_main "${TORCH_LIBRARIES}")

include(GoogleTest)
gtest_discover_tests(camera_test)
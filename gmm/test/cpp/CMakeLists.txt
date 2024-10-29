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

add_executable(multivariate_normal_test multivariate_normal_test.cpp)
target_link_libraries(multivariate_normal_test GTest::gtest_main "${TORCH_LIBRARIES}")

add_executable(spline_window_test spline_window_test.cpp)
target_link_libraries(spline_window_test GTest::gtest_main "${TORCH_LIBRARIES}")

add_executable(person_hypothesis_test person_hypothesis_test.cpp)
target_link_libraries(person_hypothesis_test GTest::gtest_main "${TORCH_LIBRARIES}")

include(GoogleTest)
gtest_discover_tests(camera_test)
gtest_discover_tests(multivariate_normal_test)
gtest_discover_tests(spline_window_test)
gtest_discover_tests(person_hypothesis_test)
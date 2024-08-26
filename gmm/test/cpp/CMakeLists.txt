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
target_link_libraries(camera_test GTest::gtest_main Eigen3::Eigen)

add_executable(multivariate_normal_test multivariate_normal_test.cpp)
target_link_libraries(multivariate_normal_test GTest::gtest_main Eigen3::Eigen)

add_executable(normal_test normal_test.cpp)
target_link_libraries(normal_test GTest::gtest_main Eigen3::Eigen)

add_executable(bspline_test bspline_test.cpp)
target_link_libraries(bspline_test GTest::gtest_main Eigen3::Eigen)

add_executable(gmm_test gmm_test.cpp)
target_link_libraries(gmm_test GTest::gtest_main Eigen3::Eigen)

add_executable(mc3d_model_test mc3d_model_test.cpp)
target_link_libraries(mc3d_model_test GTest::gtest_main Eigen3::Eigen)

add_executable(derivatives_test derivatives_test.cpp)
target_link_libraries(derivatives_test GTest::gtest_main Eigen3::Eigen)

add_executable(hypothesis_manager_test hypothesis_manager_test.cpp)
target_link_libraries(hypothesis_manager_test GTest::gtest_main Eigen3::Eigen)

include(GoogleTest)
gtest_discover_tests(camera_test)
gtest_discover_tests(multivariate_normal_test)
gtest_discover_tests(normal_test)
gtest_discover_tests(bspline_test)
gtest_discover_tests(gmm_test)
gtest_discover_tests(mc3d_model_test)
gtest_discover_tests(derivatives_test)
gtest_discover_tests(hypothesis_manager_test)
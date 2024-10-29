#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "person_hypothesis.h"

using namespace mc3d;

class PersonHypothesisTest : public PersonHypothesis
{
public:
    PersonHypothesisTest(SplineWindow &spline, const PoseParameter &pose_parameter) : PersonHypothesis(spline, pose_parameter) { }
    const Tensor get_spline_parameters() const { return spline_parameters; }
    const Tensor get_scale_factor() const { return scale_factor; }
};

TEST(PersonHypothesis, Init)
{
    Tensor knots = torch::linspace(0.0, 0.1, 5, TensorRealTypeOption.requires_grad(false));
    SplineWindow spline(knots);
    PoseParameter pose_parameter = {
        10,
        {{0, 6}, {0, 5}, {6, 5}, {6, 8}, {6, 12}, {5, 7}, {5, 11}, {8, 10}, {7, 9}, {12, 11}, {12, 14}, {11, 13}, {14, 16}, {13, 15}},
        {18.6, 18.6, 38.6, 29.9, 51.1, 29.9, 51.1, 27.5, 27.5, 20.2, 41.7, 41.7, 43.7, 43.7}
    };

    PersonHypothesisTest person_hypothesis(spline, pose_parameter);

    std::cout << "spline_parameters.sizes()" << person_hypothesis.get_spline_parameters().sizes() << std::endl;
    std::cout << "expected_spline_parameters.sizes()" << torch::IntArrayRef({10, 7, 3}) << std::endl;

    EXPECT_TRUE(person_hypothesis.get_spline_parameters().sizes() == torch::IntArrayRef({10, 7, 3}));
    EXPECT_TRUE(person_hypothesis.get_scale_factor().allclose(torch::tensor({{1.0}}, TensorRealTypeOption.requires_grad(true)).reshape({1, 1})));
}
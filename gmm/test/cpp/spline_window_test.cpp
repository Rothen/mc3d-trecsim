#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "spline_window.h"

using namespace mc3d;

class SplineWindowTest : public SplineWindow
{
public:
    SplineWindowTest(Tensor knots, size_t degree = 3, RealType lambda = 0.0, AugmentationMode augmentation_mode = AugmentationMode::Same) : SplineWindow(knots, degree, lambda, augmentation_mode) {}

    RealType basis(RealType t, int i, int k)
    {
        return SplineWindow::basis(t, i, k);
    }

    Tensor design_matrix(RealType time_point)
    {
        return SplineWindow::design_matrix(time_point);
    }

    Tensor get_knots()
    {
        return SplineWindow::knots;
    }

    size_t get_nb_basis()
    {
        return nb_basis;
    }
};

TEST(SplineWindow, Init)
{
    Tensor knots = torch::tensor({0, 1}, TensorRealTypeOption.requires_grad(false));

    SplineWindowTest spline_window(knots, 3);

    Tensor expected_knots = torch::tensor({0, 0, 0, 0, 1, 1, 1, 1}, TensorRealTypeOption.requires_grad(false));

    EXPECT_EQ(spline_window.get_nb_basis(), 4);
    EXPECT_TRUE(spline_window.get_knots().allclose(expected_knots));
}

TEST(SplineWindow, Basis)
{
    Tensor knots = torch::tensor({1, 2, 3}, TensorRealTypeOption.requires_grad(false));

    SplineWindowTest spline(knots, 3);

    EXPECT_DOUBLE_EQ(spline.basis(1.5, 2, 3), 0.25);
}

TEST(SplineWindow, DesignMatrixSameKnots)
{
    Tensor knots = torch::tensor({1, 2, 3}, TensorRealTypeOption.requires_grad(false));

    SplineWindowTest spline(knots, 3);

    Tensor times = torch::linspace(1.0, 3.0, 10, TensorRealTypeOption.requires_grad(false));

    Tensor expected_design_matrix = torch::tensor({
            {1.0, 0.0, 0.0, 0.0, 0.},
            {0.4705075445816184, 0.4636488340192044, 0.0631001371742113, 0.00274348422496571, 0.0},
            {0.1714677640603567, 0.598079561042524, 0.20850480109739367, 0.02194787379972565, 0.0},
            {0.03703703703703709, 0.5185185185185186, 0.37037037037037035, 0.07407407407407403, 0.0},
            {0.001371742112482855, 0.3401920438957476, 0.4828532235939643, 0.1755829903978052, 0.0},
            {0.0, 0.1755829903978052, 0.4828532235939643, 0.3401920438957476, 0.001371742112482855},
            {0.0, 0.07407407407407418, 0.37037037037037057, 0.5185185185185183, 0.03703703703703694},
            {0.0, 0.02194787379972568, 0.20850480109739383, 0.598079561042524, 0.17146776406035646},
            {0.0, 0.00274348422496571, 0.0631001371742113, 0.4636488340192044, 0.4705075445816184},
            {0.0, 0.0, 0.0, 0.0, 1.0}
        },
        TensorRealTypeOption.requires_grad(false));

    for (int i = 0; i < times.size(0)-1; i++)
    {
        auto design_matrix = spline.design_matrix(times[i].item().to<RealType>());
        std::cout << "Design matrix: " << design_matrix << std::endl;
        std::cout << "Expected design matrix: " << expected_design_matrix[i] << std::endl;
        ASSERT_TRUE(design_matrix.allclose(expected_design_matrix[i].reshape({1, 5})));
    }
}

TEST(SplineWindow, SplineSmoothnessLogPriorSame)
{
    Tensor knots = torch::linspace(0.0, 1.0, 10, TensorRealTypeOption.requires_grad(false));
    int degree = 3;

    SplineWindowTest spline(knots, degree, 1e-2);
    SplineParameter spline_parameter = torch::rand({static_cast<long>(spline.get_nb_basis()), 3}, TensorRealTypeOption.requires_grad(true));

    Tensor expected_spline_smoothness_log_prior = torch::tensor({
        {1.3214, -1.9821,  0.6607, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.4050, -0.6750, 0.2700, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.2202, -0.5506, 0.3303, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.4360, -1.3079, 0.8719}
    }, TensorRealTypeOption.requires_grad(false));

    std::cout << "r_spline_smoothness_log_prior: " << spline.get_smoothing_design_matrix() << std::endl;
    std::cout << "expected_spline_smoothness_log_prior: " << expected_spline_smoothness_log_prior << std::endl;

    EXPECT_TRUE(spline.get_smoothing_design_matrix().allclose(expected_spline_smoothness_log_prior, 1e-3));
}

TEST(SplineWindow, SplineSmoothnessLogPriorUniform)
{
    Tensor knots = torch::linspace(0.0, 1.0, 10, TensorRealTypeOption.requires_grad(false));
    int degree = 3;

    SplineWindowTest spline(knots, degree, 1e-2, AugmentationMode::Uniform);
    SplineParameter spline_parameter = torch::rand({static_cast<long>(spline.get_nb_basis()), 3}, TensorRealTypeOption.requires_grad(true));

    Tensor expected_spline_smoothness_log_prior = torch::tensor({
        {0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27, 0.00},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, -0.54, 0.27}
    }, TensorRealTypeOption.requires_grad(false));

    std::cout << "r_spline_smoothness_log_prior: " << spline.get_smoothing_design_matrix() << std::endl;
    std::cout << "expected_spline_smoothness_log_prior: " << expected_spline_smoothness_log_prior << std::endl;

    EXPECT_TRUE(spline.get_smoothing_design_matrix().allclose(expected_spline_smoothness_log_prior, 1e-4));
}
#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "spline_window.h"

using namespace mc3d;

class SplineWindowTest : public SplineWindow
{
public:
    SplineWindowTest(Tensor knots, size_t degree) : SplineWindow(knots, degree) {}

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

/*TEST(SplineWindow, DesignMatrixUniformKnots)
{
    Tensor knots = torch::tensor({1, 2, 3}, TensorRealTypeOption.requires_grad(false));

    SplineWindowTest spline(knots, 3);

    Tensor times = torch::linspace(1.0, 3.0, 10, TensorRealTypeOption.requires_grad(false));

    Tensor expected_design_matrix = torch::tensor({
            {1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01, 0.000000000000000000e+00, 0.000000000000000000e+00},
            {7.841792409693641719e-02, 6.227709190672153783e-01, 2.969821673525377959e-01, 1.828989483310473506e-03, 0.000000000000000000e+00},
            {2.857796067672611559e-02, 5.130315500685871388e-01, 4.437585733882029593e-01, 1.463191586648376549e-02, 0.000000000000000000e+00},
            {6.172839506172847837e-03, 3.703703703703705163e-01, 5.740740740740739589e-01, 4.938271604938268555e-02, 0.000000000000000000e+00},
            {2.286236854138091882e-04, 2.277091906721536718e-01, 6.550068587105624118e-01, 1.170553269318701239e-01, 0.000000000000000000e+00},
            {0.000000000000000000e+00, 1.170553269318701239e-01, 6.550068587105624118e-01, 2.277091906721536718e-01, 2.286236854138091882e-04},
            {0.000000000000000000e+00, 4.938271604938278270e-02, 5.740740740740741810e-01, 3.703703703703701278e-01, 6.172839506172822684e-03},
            {0.000000000000000000e+00, 1.463191586648378804e-02, 4.437585733882031258e-01, 5.130315500685870278e-01, 2.857796067672607743e-02},
            {0.000000000000000000e+00, 1.828989483310473506e-03, 2.969821673525377959e-01, 6.227709190672153783e-01, 7.841792409693641719e-02},
            {0.000000000000000000e+00, 0.000000000000000000e+00, 1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01}
        },
        TensorRealTypeOption.requires_grad(false));

    for (int i = 0; i < times.size(0); i++)
    {
        auto design_matrix = spline.design_matrix(times[i].item().to<RealType>());
        std::cout << "Design matrix: " << design_matrix << std::endl;
        std::cout << "Expected design matrix: " << expected_design_matrix[i] << std::endl;
        ASSERT_TRUE(design_matrix.allclose(expected_design_matrix[i]));
    }
}*/
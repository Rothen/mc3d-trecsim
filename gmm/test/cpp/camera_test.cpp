#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "camera.h"

using namespace mc3d;

TEST(Camera, Construct)
{
    Camera camera("camera_0");
    ASSERT_TRUE(camera.id.compare("camera_0") == 0);
}

TEST(Camera, Calibrate)
{
    Tensor A = torch::tensor({
        {1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02},
        {0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02},
        {0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00}});
    Tensor d = torch::tensor({-4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02});
    
    Tensor P = torch::tensor({
        { 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02 },
        { 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02 },
        { -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02 },
        { 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00 }
    });

    unsigned int height = 1280;
    unsigned int width = 1920;
    
    Camera camera("camera_0");
    camera.calibrate(A, d, P, height, width);

    Tensor expected_R = torch::tensor({
        {4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01},
        {4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01},
        {-7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01}
    });

    Tensor expected_R_T = torch::tensor({
        {4.894621227523461293e-01, 4.786322175197727513e-01, -7.289293729456041149e-01},
        {-4.552116929303474668e-01, 8.532140054413278607e-01, 2.545744989944053183e-01},
        {7.437803069524321353e-01, 2.072125992088085233e-01, 6.354949202935263886e-01}});

    Tensor expected_t = torch::tensor({
        { -3.875108756319725103e+02 },
        { -1.292527896409998505e+02 },
        { 1.774314430787018466e+02 }});

    std::cout << camera.t << std::endl;
    std::cout << expected_t << std::endl;

    ASSERT_TRUE(torch::equal(camera.A, A));
    ASSERT_TRUE(torch::equal(camera.d, d));
    ASSERT_TRUE(torch::equal(camera.P, P));
    ASSERT_TRUE(camera.height == height);
    ASSERT_TRUE(camera.width == width);
    ASSERT_TRUE(torch::equal(camera.R, expected_R));
    ASSERT_TRUE(torch::equal(camera.RT, expected_R_T));
    ASSERT_TRUE(torch::equal(camera.t, expected_t));
}

TEST(Camera, Transform3Dto2D)
{
    std::cout << std::fixed << std::setprecision(12);
    Tensor A = torch::tensor({{1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02},
                              {0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02},
                              {0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00}},
                             torch::dtype(torch::kDouble));

    Tensor d = torch::tensor({-4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02}, torch::dtype(torch::kDouble));

    Tensor P = torch::tensor({{4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02},
                              {4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02},
                              {-7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02},
                              {0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00}},
                             torch::dtype(torch::kDouble));

    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera camera("camera_0");
    camera.calibrate(A, d, P, height, width);

    Point3 PW = torch::tensor({100.0, -200.0, 150.0}, torch::dtype(torch::kDouble)).reshape({3, 1});

    std::cout << PW << std::endl;

    Point2 p = camera.transform_3d_to_2d(PW);

    Point2 expected_p = torch::tensor({1699.573690735038781, -408.394452051887129}, torch::dtype(torch::kDouble)).reshape({2, 1});

    std::cout << "p: " << p << std::endl;
    std::cout << "expected_p: " << expected_p << std::endl;

    EXPECT_TRUE(torch::equal(p, expected_p));
}

TEST(Camera, Transform3Dto2DGrad)
{
    Tensor A = torch::tensor({{1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02},
                              {0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02},
                              {0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00}},
                             torch::dtype(torch::kDouble));

    Tensor d = torch::tensor({-4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02}, torch::dtype(torch::kDouble));

    Tensor P = torch::tensor({{4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02},
                              {4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02},
                              {-7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02},
                              {0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00}},
                             torch::dtype(torch::kDouble));

    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera camera("camera_0");
    camera.calibrate(A, d, P, height, width);

    Point3 PW = torch::tensor({
        {100},
        {-200},
        {150}
    }, torch::dtype(torch::kDouble).requires_grad(true));

    Point2 p = camera.transform_3d_to_2d(PW);

    p[0][0].backward({}, c10::optional<bool>(true), true);

    Tensor grad_0 = torch::clone(PW.grad());

    PW.grad().zero_();

    p[1][0].backward({}, c10::optional<bool>(true), true);
    Tensor grad_1 = PW.grad();
    
    std::cout << "grad_0: " << grad_0 << std::endl;
    std::cout << "grad_1: " << grad_1 << std::endl;

    Tensor p_grad = torch::cat({grad_0, grad_1}, 1).transpose(0, 1);

    Tensor expected_p_grad = torch::tensor({{-5.619072862381639666e-02, 1.162684432384638367e+00, -3.997247651837172455e+00},
                                            {6.739469736839276415e-01, 3.561795023615820899e+00, 2.791301108698526434e+00}},
                                           torch::dtype(torch::kDouble));

    std::cout << "p_grad: " << p_grad << std::endl;
    std::cout << "expected_p_grad: " << expected_p_grad << std::endl;

    EXPECT_TRUE(torch::equal(p_grad, expected_p_grad));
}

TEST(Camera, Transform2DTo3DOffcenterCamera)
{
    Tensor A = torch::tensor({{1.137844469366448e+03, 0.0, 9.258192763436688e+02},
                              {0.0, 1.137868503272386e+03, 5.874861875982957e+02},
                              {0.0, 0.0, 1.0}},
                             torch::dtype(torch::kDouble));

    Tensor d = torch::tensor({-0.412249092780437, 0.199542410768437, -0.000678018339625, 0.00145706193774, -0.052644886162199});

    Tensor P = torch::tensor({{4.894621227523461e-01, -4.552116929303475e-01, 7.437803069524321e-01, -3.875108756319725e+02},
                              {4.786322175197728e-01, 8.532140054413279e-01, 2.072125992088085e-01, -1.292527896409999e+02},
                              {-7.289293729456041e-01, 2.545744989944053e-01, 6.354949202935264e-01, 1.774314430787018e+02},
                              {0.0, 0.0, 0.0, 1.0}},
                             torch::dtype(torch::kDouble));

    unsigned int height = 1080;
    unsigned int width = 1920;

    Camera camera("camera_0");
    camera.calibrate(A, d, P, height, width);

    Point2 p = torch::tensor({1920.0 / 2.0, 1080.0 / 2.0}, torch::dtype(torch::kDouble)).reshape({2, 1});

    std::cout << "p: " << p << std::endl;

    Point3 PW = camera.transform_2d_to_3d(p, 400.0);

    Point3 expected_PW = torch::tensor({-76.51853411744713, -54.85925460266873, 418.6210074863151},
                                       torch::dtype(torch::kDouble))
                             .reshape({3, 1});

    std::cout << "PWs: " << PW << std::endl;
    std::cout << "expectedPWs: " << expected_PW << std::endl;

    EXPECT_TRUE(torch::equal(PW, expected_PW));
}
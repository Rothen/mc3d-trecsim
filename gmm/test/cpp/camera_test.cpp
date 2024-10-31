#include "mc3d_common.h"
#include "camera.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace MC3D_TRECSIM;

TEST(Camera, Construct)
{
    Camera<double> camera("camera_0");
    ASSERT_TRUE(camera.id.compare("camera_0") == 0);
}

TEST(Camera, SetCalibration)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;
    
    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    RotationMatrix<double> expectedR;
    expectedR << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01;
    RotationMatrix<double> expectedRT;
    expectedRT << 4.894621227523461293e-01, 4.786322175197727513e-01, -7.289293729456041149e-01, -4.552116929303474668e-01, 8.532140054413278607e-01, 2.545744989944053183e-01, 7.437803069524321353e-01, 2.072125992088085233e-01, 6.354949202935263886e-01;
    TranslationVector<double> expectedt;
    expectedt << -3.875108756319725103e+02, -1.292527896409998505e+02, 1.774314430787018466e+02;

    ASSERT_TRUE(camera.A.isApprox(A));
    ASSERT_TRUE(camera.d.isApprox(d));
    ASSERT_TRUE(camera.P.isApprox(P));
    ASSERT_TRUE(camera.height == height);
    ASSERT_TRUE(camera.width == width);
    ASSERT_TRUE(camera.R.isApprox(expectedR));
    ASSERT_TRUE(camera.RT.isApprox(expectedRT));
    ASSERT_TRUE(camera.t.isApprox(expectedt));
}

TEST(Camera, ToCameraCoordinates)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoints<double> PWs(3, 1);
    PWs << 100, -200, 150;

    WorldPoints<double> PCs = camera.toCameraCoordinates(PWs);

    WorldPoints<double> expectedPCs(3, 1);
    expectedPCs << 2.247517984765653694e+02, -2.892665076210431039e+02, 3.305087326421007106e+02;

    EXPECT_TRUE(PCs.isApprox(expectedPCs));
}

TEST(Camera, ToCameraCoordinatesMeters)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+00, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+00, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoints<double> PWs(3, 1);
    PWs << 1.00, -2.00, 1.50;

    WorldPoints<double> PCs = camera.toCameraCoordinates(PWs);

    WorldPoints<double> expectedPCs(3, 1);
    expectedPCs << 2.247517984765653694e+00, -2.892665076210431039e+00, 3.305087326421007106e+00;

    EXPECT_TRUE(PCs.isApprox(expectedPCs));
}

TEST(Camera, ToCameraCoordinatesSinglePoint)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoint<double> PW(3);
    PW << 100, -200, 150;

    WorldPoint<double> PC = camera.toCameraCoordinates(PW);

    WorldPoint<double> expectedPC(3);
    expectedPC << 2.247517984765653694e+02, -2.892665076210431039e+02, 3.305087326421007106e+02;

    EXPECT_TRUE(PC.isApprox(expectedPC));
}

TEST(Camera, Project)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoints<double> PWs(3, 1);
    PWs << 100, -200, 150;

    std::cout << PWs << std::endl;

    CameraPoints<double> ps = camera.project(PWs);

    CameraPoints<double> expectedps(2, 1);
    expectedps << 1.699573690735038781e+03, -4.083944520518871286e+02;

    EXPECT_TRUE(ps.isApprox(expectedps));
}

TEST(Camera, ProjectMeters)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+00, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+00, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoints<double> PWs(3, 1);
    PWs << 1.00, -2.00, 1.50;

    std::cout << PWs << std::endl;

    CameraPoints<double> ps = camera.project(PWs);

    CameraPoints<double> expectedps(2, 1);
    expectedps << 1.699573690735038781e+03, -4.083944520518871286e+02;

    EXPECT_TRUE(ps.isApprox(expectedps));
}

TEST(Camera, ProjectGrad)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoint<double> PW;
    PW << 100, -200, 150;

    std::cout << PW << std::endl;

    CameraPointGrad<double> pGrad = camera.projectGrad(PW);

    CameraPointGrad<double> expectedpGrad(2, 3);
    expectedpGrad << -5.619072862381639666e-02, 1.162684432384638367e+00, -3.997247651837172455e+00, 6.739469736839276415e-01, 3.561795023615820899e+00, 2.791301108698526434e+00;

    EXPECT_TRUE(pGrad.isApprox(expectedpGrad));
}

TEST(Camera, ProjectGradMeters)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+00, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+00, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    WorldPoint<double> PW;
    PW << 1.00, -2.00, 1.50;

    std::cout << PW << std::endl;

    CameraPointGrad<double> pGrad = camera.projectGrad(PW);

    CameraPointGrad<double> expectedpGrad(2, 3);
    expectedpGrad << -5.619072862381639666e-02, 1.162684432384638367e+00, -3.997247651837172455e+00, 6.739469736839276415e-01, 3.561795023615820899e+00, 2.791301108698526434e+00;

    std::cout << "pGrad: " << pGrad << std::endl;
    std::cout << "expectedpGrad: " << expectedpGrad << std::endl;

    EXPECT_TRUE(pGrad.isApprox(expectedpGrad));
}

TEST(Camera, pixelsToWorldPoints)
{
    IntrinsicMatrix<double> A;
    A << 1.123478282832530e+03, 0.0, 9.624201914371442e+02, 0.0, 1.115995046593397e+03, 5.875233741561747e+02, 0.0, 0.0, 1.0;
    DistortionVector<double> d;
    d << -4.098335442199602e-01, 2.065881602949584e-01, -1.852675935335392e-03, -1.274891630199873e-04, -6.102605512197318e-02;
    ExtrinsicMatrix<double> P;
    P << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    unsigned int height = 1080;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    CameraPoints<double> pIs(2, 1);
    pIs << 1920.0/2.0, 1080.0/2.0;

    std::cout << "pIs: " << pIs << std::endl;

    WorldPoints<double> PWs = camera.pixelsToWorldPoints(pIs, 400.0);

    WorldPoints<double> expectedPWs(3, 1);
    expectedPWs << -0.8616780490112, -17.033543043489672, 400.0;

    std::cout << "PWs: " << PWs << std::endl;
    std::cout << "expectedPWs: " << expectedPWs << std::endl;

    EXPECT_TRUE(PWs.isApprox(expectedPWs));
}

TEST(Camera, pixelsToWorldPointsOffcenterCamera)
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448e+03, 0.0, 9.258192763436688e+02, 0.0, 1.137868503272386e+03, 5.874861875982957e+02, 0.0, 0.0, 1.0;
    DistortionVector<double> d;
    d << -0.412249092780437, 0.199542410768437, -0.000678018339625, 0.00145706193774, -0.052644886162199;
    ExtrinsicMatrix<double> P;
    P << 4.894621227523461e-01, -4.552116929303475e-01, 7.437803069524321e-01, -3.875108756319725e+02,
        4.786322175197728e-01, 8.532140054413279e-01, 2.072125992088085e-01, -1.292527896409999e+02,
        -7.289293729456041e-01, 2.545744989944053e-01, 6.354949202935264e-01, 1.774314430787018e+02,
        0.0, 0.0, 0.0, 1.0;
    unsigned int height = 1080;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    CameraPoints<double> pIs(2, 1);
    pIs << 1920.0 / 2.0, 1080.0 / 2.0;

    std::cout << "pIs: " << pIs << std::endl;

    WorldPoints<double> PWs = camera.pixelsToWorldPoints(pIs, 400.0);

    WorldPoints<double> expectedPWs(3, 1);
    expectedPWs << -76.51853411744713, -54.85925460266873, 418.6210074863151;

    std::cout << "PWs: " << PWs << std::endl;
    std::cout << "expectedPWs: " << expectedPWs << std::endl;

    EXPECT_TRUE(PWs.isApprox(expectedPWs));
}

TEST(Camera, isPointInFrame)
{
    IntrinsicMatrix<double> A;
    A << 1.123478282832530e+03, 0.0, 9.624201914371442e+02, 0.0, 1.115995046593397e+03, 5.875233741561747e+02, 0.0, 0.0, 1.0;
    DistortionVector<double> d;
    d << -4.098335442199602e-01, 2.065881602949584e-01, -1.852675935335392e-03, -1.274891630199873e-04, -6.102605512197318e-02;
    ExtrinsicMatrix<double> P;
    P << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    unsigned int height = 1080;
    unsigned int width = 1920;

    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);

    CameraPoint<double> pI(2);

    pI << 0, 0;
    ASSERT_TRUE(camera.isPointInFrame(pI));
    pI << 0, 1080.0;
    ASSERT_TRUE(camera.isPointInFrame(pI));
    pI << 1920.0, 0;
    ASSERT_TRUE(camera.isPointInFrame(pI));
    pI << 1920.0, 1080.0;
    ASSERT_TRUE(camera.isPointInFrame(pI));
    pI << 1920.0 / 2.0, 1080.0 / 2.0;
    ASSERT_TRUE(camera.isPointInFrame(pI));
    pI << -1, 0;
    ASSERT_FALSE(camera.isPointInFrame(pI));
    pI << 0, -1;
    ASSERT_FALSE(camera.isPointInFrame(pI));
    pI << 1921.0, 0;
    ASSERT_FALSE(camera.isPointInFrame(pI));
    pI << 0, 1081.0;
    ASSERT_FALSE(camera.isPointInFrame(pI));
    pI << 1921.0, 1081.0;
    ASSERT_FALSE(camera.isPointInFrame(pI));
}
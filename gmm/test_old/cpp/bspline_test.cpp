#include "bspline.h"
#include "mc3d_common.h"
#include "test_shared.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace MC3D_TRECSIM;

TEST(BSpline, UniformKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, UNIFORM, knots);

    Vector<double> expectedKnots(9);
    expectedKnots << -2, -1, 0, 1, 2, 3, 4, 5, 6;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, SameKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, SAME, knots);

    Vector<double> expectedKnots(9);
    expectedKnots << 1, 1, 1, 1, 2, 3, 3, 3, 3;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, NoneKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, NONE, knots);

    Vector<double> expectedKnots(3);
    expectedKnots << 1, 2, 3;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, Basis)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, UNIFORM, knots);

    auto tstart = std::chrono::steady_clock::now();
    spline.basis(1.5, 2, 3);
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << cputime.count() << std::endl;

    ASSERT_TRUE(true);
}

TEST(BSpline, CreateDesignMatrix)
{
    Vector<double> knots(3);
    knots << 1.0, 2.0, 3.0;

    BSpline<double> spline(3, UNIFORM, knots);

    Vector<double> times = Vector<double>::LinSpaced(10, 1.0, 3.0);

    auto tstart = std::chrono::steady_clock::now();

    auto designMatrix = spline.designMatrix(times);

    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "CPU Time First:" << cputime.count() << std::endl;

    tstart = std::chrono::steady_clock::now();

    designMatrix = spline.designMatrix(times);

    tend = std::chrono::steady_clock::now();
    cputime = tend - tstart;
    std::cout << "CPU Time Repeated:" << cputime.count() << std::endl;

    RowMatrix<double> expectedDesignMatrix(10, spline.getNumBasis());
    expectedDesignMatrix << 1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 7.841792409693641719e-02, 6.227709190672153783e-01, 2.969821673525377959e-01, 1.828989483310473506e-03, 0.000000000000000000e+00, 2.857796067672611559e-02, 5.130315500685871388e-01, 4.437585733882029593e-01, 1.463191586648376549e-02, 0.000000000000000000e+00, 6.172839506172847837e-03, 3.703703703703705163e-01, 5.740740740740739589e-01, 4.938271604938268555e-02, 0.000000000000000000e+00, 2.286236854138091882e-04, 2.277091906721536718e-01, 6.550068587105624118e-01, 1.170553269318701239e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.170553269318701239e-01, 6.550068587105624118e-01, 2.277091906721536718e-01, 2.286236854138091882e-04, 0.000000000000000000e+00, 4.938271604938278270e-02, 5.740740740740741810e-01, 3.703703703703701278e-01, 6.172839506172822684e-03, 0.000000000000000000e+00, 1.463191586648378804e-02, 4.437585733882031258e-01, 5.130315500685870278e-01, 2.857796067672607743e-02, 0.000000000000000000e+00, 1.828989483310473506e-03, 2.969821673525377959e-01, 6.227709190672153783e-01, 7.841792409693641719e-02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01;

    ASSERT_TRUE(designMatrix.isApprox(expectedDesignMatrix));
}

TEST(BSpline, PassDesignMatrix)
{
    Vector<double> knots(3);
    knots << 1.0, 2.0, 3.0;

    BSpline<double> spline(3, UNIFORM, knots);

    Vector<double> times = Vector<double>::LinSpaced(10, 1.0, 3.0);
    RowMatrix<double> A = RowMatrix<double>::Zero(times.size(), spline.getNumBasis());

    auto tstart = std::chrono::steady_clock::now();

    spline.designMatrix(times, A);

    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "CPU Time First:" << cputime.count() << std::endl;

    tstart = std::chrono::steady_clock::now();

    spline.designMatrix(times, A);

    tend = std::chrono::steady_clock::now();
    cputime = tend - tstart;
    std::cout << "CPU Time Repeated:" << cputime.count() << std::endl;

    RowMatrix<double> expectedA(10, spline.getNumBasis());
    expectedA << 1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 7.841792409693641719e-02, 6.227709190672153783e-01, 2.969821673525377959e-01, 1.828989483310473506e-03, 0.000000000000000000e+00, 2.857796067672611559e-02, 5.130315500685871388e-01, 4.437585733882029593e-01, 1.463191586648376549e-02, 0.000000000000000000e+00, 6.172839506172847837e-03, 3.703703703703705163e-01, 5.740740740740739589e-01, 4.938271604938268555e-02, 0.000000000000000000e+00, 2.286236854138091882e-04, 2.277091906721536718e-01, 6.550068587105624118e-01, 1.170553269318701239e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.170553269318701239e-01, 6.550068587105624118e-01, 2.277091906721536718e-01, 2.286236854138091882e-04, 0.000000000000000000e+00, 4.938271604938278270e-02, 5.740740740740741810e-01, 3.703703703703701278e-01, 6.172839506172822684e-03, 0.000000000000000000e+00, 1.463191586648378804e-02, 4.437585733882031258e-01, 5.130315500685870278e-01, 2.857796067672607743e-02, 0.000000000000000000e+00, 1.828989483310473506e-03, 2.969821673525377959e-01, 6.227709190672153783e-01, 7.841792409693641719e-02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.666666666666666574e-01, 6.666666666666666297e-01, 1.666666666666666574e-01;

    ASSERT_TRUE(A.isApprox(expectedA));
}

TEST(BSpline, PushUniformKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, UNIFORM, knots);
    spline.pushKnot(5);

    Vector<double> expectedKnots(10);
    expectedKnots << -2, -1, 0, 1, 2, 3, 5, 7, 9, 11;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
    spline.pushKnot(6);

    Vector<double> secondExpectedKnots(11);
    secondExpectedKnots << -2, -1, 0, 1, 2, 3, 5, 6, 7, 8, 9;

    ASSERT_TRUE(spline.getKnots().isApprox(secondExpectedKnots));
}

TEST(BSpline, PushSameKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, SAME, knots);
    spline.pushKnot(4);

    Vector<double> expectedKnots(10);
    expectedKnots << 1, 1, 1, 1, 2, 3, 4, 4, 4, 4;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, PushNoneKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, NONE, knots);
    spline.pushKnot(4);

    Vector<double> expectedKnots(4);
    expectedKnots << 1, 2, 3, 4;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, PopFrontUniformKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, UNIFORM, knots);
    spline.popKnotFront();

    Vector<double> expectedKnots(8);
    expectedKnots << -1, 0, 1, 2, 3, 4, 5, 6;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, PopFrontSameKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, SAME, knots);
    spline.popKnotFront();

    Vector<double> expectedKnots(8);
    expectedKnots << 2, 2, 2, 2, 3, 3, 3, 3;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, PopFrontNoneKnotAugmentation)
{
    Vector<double> knots(3);
    knots << 1, 2, 3;

    BSpline<double> spline(3, NONE, knots);
    spline.popKnotFront();

    Vector<double> expectedKnots(2);
    expectedKnots << 2, 3;

    ASSERT_TRUE(spline.getKnots().isApprox(expectedKnots));
}

TEST(BSpline, SmoothDesignMatrix)
{
    Vector<double> knots = Vector<double>::LinSpaced(10, 0.0, 1.0);
    Vector<double> x(20);
    x << 0.0, 0.02, 0.06, 0.14, 0.17, 0.19, 0.22, 0.27, 0.33, 0.40, 0.47, 0.52, 0.56, 0.60, 0.62, 0.67, 0.75, 0.84, 0.92, 1.0;
    int degree = 3;

    BSpline<double> spline(degree, UNIFORM, knots);

    auto designMatrix = spline.designMatrix(x);
    std::cout << designMatrix << std::endl;
    spline.smoothDesignMatrix(designMatrix, 1e-2);
    std::cout << designMatrix.format(HeavyFormat) << std::endl;
    std::cout << "shape: (" << designMatrix.rows() << ", " << designMatrix.cols() << ")" << std::endl;

    RowMatrix<double> expectedSmoothedDesignMatrix(x.size() + spline.getNumBasis() - 2, spline.getNumBasis());
    expectedSmoothedDesignMatrix << 0.166666666666667, 0.666666666666667, 0.166666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0918946666666667, 0.637182666666667, 0.269950666666667, 0.000972, 0, 0, 0, 0, 0, 0, 0, 0, 0.0162226666666667, 0.453798666666667, 0.503734666666667, 0.026244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0675373333333333, 0.607854666666667, 0.321678666666667, 0.00292933333333334, 0, 0, 0, 0, 0, 0, 0, 0, 0.0173038333333333, 0.460205166666667, 0.497678166666667, 0.0248128333333334, 0, 0, 0, 0, 0, 0, 0, 0, 0.00406483333333333, 0.341522166666667, 0.594761166666667, 0.0596518333333334, 0, 0, 0, 0, 0, 0, 0, 0, 1.33333333333331e-06, 0.176862666666667, 0.666270666666667, 0.156865333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0308655, 0.521520166666666, 0.434363166666667, 0.0132511666666667, 0, 0, 0, 0, 0, 0, 0, 0, 4.49999999999986e-06, 0.182103166666667, 0.665780166666667, 0.152112166666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0106666666666666, 0.414666666666666, 0.538666666666667, 0.0360000000000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0760888333333334, 0.619850166666667, 0.302033166666667, 0.00202783333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0.00546133333333333, 0.361482666666666, 0.580650666666667, 0.0524053333333334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.147456, 0.665098666666667, 0.187434666666667, 1.06666666666669e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.036, 0.538666666666667, 0.414666666666667, 0.0106666666666666, 0, 0, 0, 0, 0, 0, 0, 0, 0.012348, 0.427822666666667, 0.527310666666667, 0.0325186666666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.152112166666666, 0.665780166666667, 0.182103166666667, 4.50000000000031e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0.00260416666666664, 0.315104166666666, 0.611979166666667, 0.0703125000000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0141973333333333, 0.440874666666667, 0.515658666666667, 0.0292693333333334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0622079999999998, 0.599242666666666, 0.334890666666667, 0.00365866666666669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.166666666666667, 0.666666666666667, 0.166666666666667, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27, -0.54, 0.27;
    ASSERT_TRUE(designMatrix.isApprox(expectedSmoothedDesignMatrix));
}

TEST(BSpline, SmoothDesignMatrixWithThetaEqZero)
{
    Vector<double> knots = Vector<double>::LinSpaced(10, 0.0, 1.0);
    Vector<double> x(20);
    x << 0.0, 0.02, 0.06, 0.14, 0.17, 0.19, 0.22, 0.27, 0.33, 0.40, 0.47, 0.52, 0.56, 0.60, 0.62, 0.67, 0.75, 0.84, 0.92, 1.0;
    int degree = 3;

    BSpline<double> spline(degree, UNIFORM, knots);

    auto designMatrix = spline.designMatrix(x);
    std::cout << designMatrix << std::endl;
    spline.smoothDesignMatrix(designMatrix, 0);
    std::cout << designMatrix << std::endl;
    std::cout << "shape: (" << designMatrix.rows() << ", " << designMatrix.cols() << ")" << std::endl;

    RowMatrix<double> expectedSmoothedDesignMatrix(x.size() + spline.getNumBasis() - 2, spline.getNumBasis());
    expectedSmoothedDesignMatrix << 1.666666666666667e-01, 6.666666666666666e-01, 1.666666666666667e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.189466666666665e-02, 6.371826666666666e-01, 2.699506666666667e-01, 9.720000000000002e-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.622266666666666e-02, 4.537986666666666e-01, 5.037346666666667e-01, 2.624400000000000e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.753733333333327e-02, 6.078546666666665e-01, 3.216786666666668e-01, 2.929333333333339e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.730383333333331e-02, 4.602051666666666e-01, 4.976781666666669e-01, 2.481283333333336e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.064833333333328e-03, 3.415221666666666e-01, 5.947611666666667e-01, 5.965183333333336e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.333333333333309e-06, 1.768626666666666e-01, 6.662706666666667e-01, 1.568653333333334e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.086549999999995e-02, 5.215201666666665e-01, 4.343631666666669e-01, 1.325116666666669e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.499999999999862e-06, 1.821031666666665e-01, 6.657801666666667e-01, 1.521121666666668e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.066666666666663e-02, 4.146666666666665e-01, 5.386666666666667e-01, 3.600000000000006e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.608883333333341e-02, 6.198501666666667e-01, 3.020331666666665e-01, 2.027833333333332e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.461333333333332e-03, 3.614826666666665e-01, 5.806506666666668e-01, 5.240533333333341e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.474559999999997e-01, 6.650986666666665e-01, 1.874346666666670e-01, 1.066666666666689e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.600000000000000e-02, 5.386666666666666e-01, 4.146666666666668e-01, 1.066666666666665e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.234799999999998e-02, 4.278226666666665e-01, 5.273106666666669e-01, 3.251866666666667e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.521121666666663e-01, 6.657801666666667e-01, 1.821031666666670e-01, 4.500000000000314e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.604166666666642e-03, 3.151041666666664e-01, 6.119791666666669e-01, 7.031250000000012e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.419733333333331e-02, 4.408746666666666e-01, 5.156586666666667e-01, 2.926933333333339e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.220799999999985e-02, 5.992426666666664e-01, 3.348906666666671e-01, 3.658666666666694e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.666666666666667e-01, 6.666666666666666e-01, 1.666666666666667e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    ASSERT_TRUE(designMatrix.isApprox(expectedSmoothedDesignMatrix));
}
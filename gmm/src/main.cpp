#include "config.h"
#include "mc3d_common.h"
#include "frame.h"
#include "camera.h"
#include "gmm_param.h"
#include "gmm.h"
#include "json.hpp"

#include <iostream>
#include <fstream>
#include <LBFGS.h>
#include <chrono>

// For convenience
using json = nlohmann::json;

template <typename T>
void fillupEigenMatrix(const std::vector<T> &source, MC3D_TRECSIM::RowMatrix<T> &dest)
{
    for (int r{0}; r < dest.rows(); ++r)
    {
        for (int c{0}; c < dest.cols(); ++c)
        {
            dest(r, c) = source[r * dest.cols() + c];
        }
    }
}

template <typename T>
void fillupEigenVector(const std::vector<T> &source, MC3D_TRECSIM::Vector<T> &dest)
{
    for (int i{0}; i < dest.size(); ++i)
    {
        dest[i] = source[i];
    }
}

bool parseJsonFile(const std::string &filename, std::vector<MC3D_TRECSIM::Frame<double>> &frames, std::vector<MC3D_TRECSIM::Camera<double>> &cameras)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return false;
    }

    json j;
    try
    {
        file >> j;
        const int rows{17};
        const int cols{3};

        for (const auto &item : j["frames"].items())
        {
            const int cameraIndex = item.value()["cameraIndex"].template get<int>();
            const double time = item.value()["time"].template get<double>();
            const double origTimestamp = item.value()["origTimestamp"].template get<double>();
            const std::vector<std::vector<double>> jsonKpts = item.value()["kpts"].template get<std::vector<std::vector<double>>>();

            std::vector<MC3D_TRECSIM::RowMatrix<double>> kpts;

            for (const std::vector<double> &jsonPerson : jsonKpts)
            {
                MC3D_TRECSIM::RowMatrix<double> person(rows, cols);
                fillupEigenMatrix(jsonPerson, person);

                kpts.push_back(person);
            }

            frames.push_back(MC3D_TRECSIM::Frame<double>(cameraIndex, kpts, time, origTimestamp));
        }

        for (const auto &item : j["cameras"].items())
        {
            const std::string id = item.value()["id"].template get<std::string>();
            const int height = item.value()["height"].template get<int>();
            const int width = item.value()["width"].template get<int>();
            const float distance = item.value()["distance"].template get<float>();
            const std::vector<double> jsonA = item.value()["A"].template get<std::vector<double>>();
            const std::vector<double> jsond = item.value()["d"].template get<std::vector<double>>();
            const std::vector<double> jsonP = item.value()["P"].template get<std::vector<double>>();

            MC3D_TRECSIM::RowMatrix<double> A(3, 3);
            MC3D_TRECSIM::Vector<double> d(5);
            MC3D_TRECSIM::RowMatrix<double> P(4, 4);

            fillupEigenMatrix(jsonA, A);
            fillupEigenVector(jsond, d);
            fillupEigenMatrix(jsonP, P);

            cameras.push_back(MC3D_TRECSIM::Camera<double>(id, A, d, P, height, width, distance));
        }
        return true;
    }
    catch (json::parse_error &e)
    {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

MC3D_TRECSIM::GMMParam<double> getGMMParam()
{
    MC3D_TRECSIM::GMMParam<double> gmmParam;
    gmmParam.tol = 1e-6;
    gmmParam.splineKnotDelta = 200;
    gmmParam.maxFrameBuffer = 20;
    gmmParam.nu = 500;
    gmmParam.autoManageTheta = true;
    gmmParam.autoManageHypothesis = true;
    gmmParam.copyLastThetas = true;
    gmmParam.splineSmoothingFactor = 10;
    gmmParam.maxIter = 5;
    return gmmParam;
}

LBFGSpp::LBFGSParam<double> getLBFGSParam()
{
    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 20;
    lbfgsParam.max_linesearch = 5;
    return lbfgsParam;
}

int start(const std::string &filename)
{
    std::vector<MC3D_TRECSIM::Frame<double>> frames;
    std::vector<MC3D_TRECSIM::Camera<double>> cameras;

    if (!parseJsonFile(filename, frames, cameras))
    {
        return 1;
    }

    std::vector<int> KEYPOINTS{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    MC3D_TRECSIM::GMMParam<double> gmmParam = getGMMParam();
    gmmParam.KEYPOINTS = KEYPOINTS;

    LBFGSpp::LBFGSParam<double> lbfgsParam = getLBFGSParam();

    MC3D_TRECSIM::GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    int startFrame{0};
    int endFrame{90};

    for (int i{startFrame}; i <= endFrame; ++i)
    {
        gmm.addFrame(frames[i]);
    }

    std::cout << "# Basis: " << gmm.spline.getNumBasis() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto fitResult = gmm.fit();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "# iterations: [";
    for (int i = 0; i < fitResult[5].niters.size(); i++)
    {
        std::cout << fitResult[5].niters[i] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Time needed: " << double(duration.count())/1000.0 << "ms" << std::endl;

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    return start(filename);
}
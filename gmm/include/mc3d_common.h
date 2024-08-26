#ifndef COMMON_H
#define COMMON_H

#include "config.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <memory>
#include <Eigen/Dense>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    using ColMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename Scalar>
    using RowMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    template <typename Scalar>
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

    template <typename Scalar>
    using IntrinsicMatrix = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;
    template <typename Scalar>
    using DistortionVector = Eigen::Vector<Scalar, 5>;
    template <typename Scalar>
    using ExtrinsicMatrix = Eigen::Matrix<Scalar, 4, 4, Eigen::RowMajor>;
    template <typename Scalar>
    using RotationMatrix = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;
    template <typename Scalar>
    using TranslationVector = Eigen::Vector<Scalar, 3>;
    template <typename Scalar>
    using WorldPoint = Eigen::Vector<Scalar, 3>;
    template <typename Scalar>
    using WorldPoints = Eigen::Matrix<Scalar, 3, Eigen::Dynamic, Eigen::ColMajor>;
    template <typename Scalar>
    using CameraPoint = Eigen::Vector<Scalar, 2>;
    template <typename Scalar>
    using CameraPoints = Eigen::Matrix<Scalar, 2, Eigen::Dynamic, Eigen::ColMajor>;
    template <typename Scalar>
    using CameraPointGrad = Eigen::Matrix<Scalar, 2, 3, Eigen::RowMajor>;

    template <typename Scalar>
    void logEigen(std::shared_ptr<spdlog::logger> &logger, Eigen::Vector<Scalar, Eigen::Dynamic> &v, const char prefix[] = "")
    {
        Eigen::IOFormat HeavyFormat(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::stringstream ss;
        ss << prefix << v.transpose().format(HeavyFormat);
        logger->info(ss.str());
    }

    template <typename Scalar>
    void logEigen(std::shared_ptr<spdlog::logger> &logger, RowMatrix<Scalar> &M, const char prefix[] = "")
    {
        Eigen::IOFormat HeavyFormat(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::stringstream ss;
        ss << prefix << M.reshaped(M.rows() * M.cols(), 1).format(HeavyFormat);
        logger->info(ss.str());
    }

    template <typename Scalar>
    void logEigen(std::shared_ptr<spdlog::logger> &logger, ColMatrix<Scalar> &M, const char prefix[] = "")
    {
        std::stringstream ss;
        Eigen::IOFormat HeavyFormat(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

        ss << prefix << M.transpose().reshaped(M.rows() * M.cols(), 1).format(HeavyFormat);
        logger->info(ss.str());
    }
}

#endif
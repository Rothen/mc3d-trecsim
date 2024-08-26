#ifndef CAMERA_H
#define CAMERA_H

#include "config.h"
#include "mc3d_common.h"
#include "bspline.h"

#include <Eigen/Core>
#include <string>
#include <limits>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class Camera
    {
    public:
        std::string id;
        unsigned int height;
        unsigned int width;
        Scalar distance;
        // Intrinsic camera matrix
        IntrinsicMatrix<Scalar> A;
        // Intrinsic camera matrix
        IntrinsicMatrix<Scalar> Ainv;
        // Camera distortion vector
        DistortionVector<Scalar> d;
        // Extrinsic camera matrix
        ExtrinsicMatrix<Scalar> P;
        // Rotation matrix part of the extrinsic matrix
        RotationMatrix<Scalar> R;
        // Transpose of the rotation matrix (also the inverse)
        RotationMatrix<Scalar> RT;
        // Translation vector part of the extrinsic matrix
        TranslationVector<Scalar> t;

        Camera(std::string id = "", IntrinsicMatrix<Scalar> A = IntrinsicMatrix<Scalar>::Identity(3, 3), DistortionVector<Scalar> d = DistortionVector<Scalar>::Zero(5), ExtrinsicMatrix<Scalar> P = ExtrinsicMatrix<Scalar>::Identity(4, 4), unsigned int height = 0, unsigned int width = 0, Scalar distance = 1.0);

        inline void setCalibration(IntrinsicMatrix<Scalar> A, DistortionVector<Scalar> d, ExtrinsicMatrix<Scalar> P, unsigned int height, unsigned int width, Scalar distance = 1.0);

        inline WorldPoints<Scalar> toCameraCoordinates(const WorldPoints<Scalar> &PWs) const;

        inline WorldPoints<Scalar> pixelsToWorldPoints(const CameraPoints<Scalar> &pIs, Scalar distance) const;

        inline WorldPoints<Scalar> pixelsToWorldPoints(const CameraPoints<Scalar> &pIs) const;

        inline CameraPoints<Scalar> project(const WorldPoints<Scalar> &PWs) const;

        inline CameraPoint<Scalar> projectSingle(const WorldPoint<Scalar> &PW) const;

        inline CameraPointGrad<Scalar> projectGrad(const WorldPoint<Scalar> &PW);

        inline void projectSingle(const WorldPoint<Scalar> &PW, CameraPoint<Scalar> &dest) const;

        inline void transformWorldCenter(const ExtrinsicMatrix<Scalar> &P);

        inline void projectGrad(const WorldPoint<Scalar> &PW, CameraPointGrad<Scalar> &dest);

        inline bool isPointInFrame(const CameraPoint<Scalar> &pI) const;

    private:
        const WorldPoint<Scalar> limiter{0.0, 0.0, std::numeric_limits<Scalar>::epsilon()};

        WorldPoint<Scalar> tempPC;
        Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> tempDP;
    };
}
#include "camera_impl.h"
#endif
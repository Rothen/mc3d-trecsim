#ifndef CAMERA_IMPL_H
#define CAMERA_IMPL_H

#include "camera.h"

#include <Eigen/Core>
#include <string>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    Camera<Scalar>::Camera(std::string id, IntrinsicMatrix<Scalar> A, DistortionVector<Scalar> d, ExtrinsicMatrix<Scalar> P, unsigned int height, unsigned int width, Scalar distance) :
        id(id)
    {
        setCalibration(std::move(A), std::move(d), std::move(P), height, width);
    }

    template <typename Scalar>
    inline void Camera<Scalar>::setCalibration(IntrinsicMatrix<Scalar> A, DistortionVector<Scalar> d, ExtrinsicMatrix<Scalar> P, unsigned int height, unsigned int width, Scalar distance)
    {
        this->A = std::move(A);
        this->d = std::move(d);
        this->P = std::move(P);
        this->height = height;
        this->width = width;
        this->distance = distance;

        Ainv = A.inverse();
        R = P.topLeftCorner(3, 3);
        RT = R.transpose();
        t = P.col(3).head(3);
    }

    template <typename Scalar>
    inline WorldPoints<Scalar> Camera<Scalar>::toCameraCoordinates(const WorldPoints<Scalar> &PWs) const
    {
        WorldPoints<Scalar> PCs = RT * (PWs - t);

        for (int c = 0; c < PCs.cols(); c++)
        {
            if (PCs.col(c)[2] == 0)
            {
                PCs.col(c) << 0.0, 0.0, 1.0;
            }
        }

        return PCs; // (RT * (PWs - t)).colwise() + limiter;
    }

    template <typename Scalar>
    inline WorldPoints<Scalar> Camera<Scalar>::pixelsToWorldPoints(const CameraPoints<Scalar> &pIs, Scalar distance) const
    {
        WorldPoints<Scalar> newPIs = WorldPoints<Scalar>::Ones(3, pIs.cols());
        newPIs.topRows(2) = pIs;
        return R * ((Ainv * newPIs) * distance) + t;
    }

    template <typename Scalar>
    inline WorldPoints<Scalar> Camera<Scalar>::pixelsToWorldPoints(const CameraPoints<Scalar> &pIs) const
    {
        return pixelsToWorldPoints(pIs, distance);
    }

    template <typename Scalar>
    inline CameraPoints<Scalar> Camera<Scalar>::project(const WorldPoints<Scalar> &PWs) const
    {
        return (A * toCameraCoordinates(PWs).colwise().hnormalized().colwise().homogeneous()).topRows(2);
    }

    template <typename Scalar>
    inline void Camera<Scalar>::projectSingle(const WorldPoint<Scalar> &PW, CameraPoint<Scalar> &dest) const
    {
        dest << (A * (RT * (PW - t)).hnormalized().homogeneous()).head(2);
    }

    template <typename Scalar>
    inline CameraPoint<Scalar> Camera<Scalar>::projectSingle(const WorldPoint<Scalar> &PW) const
    {
        CameraPoint<Scalar> dest;
        projectSingle(PW, dest);
        return dest;
    }

    template <typename Scalar>
    inline void Camera<Scalar>::projectGrad(const WorldPoint<Scalar> &PW, CameraPointGrad<Scalar> &dest)
    {
        tempPC << RT * (PW - t);

        tempDP << (1.0 / tempPC[2]), 0, (-tempPC[0] / (tempPC[2] * tempPC[2])),
            0.0, (1.0 / tempPC[2]), (-tempPC[1] / (tempPC[2] * tempPC[2])),
            0.0, 0.0, 0.0;

        dest << (A * tempDP * RT).topRows(2) / 100.0;
    }

    template <typename Scalar>
    inline CameraPointGrad<Scalar> Camera<Scalar>::projectGrad(const WorldPoint<Scalar> &PW)
    {
        CameraPointGrad<Scalar> dest;
        projectGrad(PW, dest);
        return dest;
    }

    template <typename Scalar>
    inline void Camera<Scalar>::transformWorldCenter(const ExtrinsicMatrix<Scalar> &P)
    {
        this->P = P * this->P;
        setCalibration(A, d, this->P, height, width);
    }

    template <typename Scalar>
    inline bool Camera<Scalar>::isPointInFrame(const CameraPoint<Scalar> &pI) const {
        return pI[0] >= 0 && pI[0] <= width && pI[1] >= 0 && pI[1] <= height;
    }
};
#endif
#ifndef CAMERA_H
#define CAMERA_H

#include "mc3d_common.h"

namespace mc3d
{
    class Camera
    {
    public:
        std::string id;
        unsigned int height;
        unsigned int width;
        RealType distance;
        // Intrinsic camera matrix
        Tensor A;
        // Intrinsic camera matrix
        Tensor Ainv;
        // Camera distortion vector
        Tensor d;
        // Extrinsic camera matrix
        Tensor P;
        // Rotation matrix part of the extrinsic matrix
        Tensor R;
        // Transpose of the rotation matrix (also the inverse)
        Tensor RT;
        // Translation vector part of the extrinsic matrix
        Tensor t;

        Camera(std::string id = "", Tensor A = torch::eye(3), Tensor d = torch::zeros(5), Tensor P = torch::eye(4), unsigned int height = 0, unsigned int width = 0, RealType distance = 1.0);
        void calibrate(Tensor A, Tensor d, Tensor P, unsigned int height, unsigned int width, RealType distance = 1.0);
        Point2 transform_3d_to_2d(Point3 p) const;
        Point3 transform_2d_to_3d(Point2 P, RealType distance) const;

    protected:
    };
}

#include "camera_impl.h"

#endif
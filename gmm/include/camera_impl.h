#ifndef CAMERA_IMPL_H
#define CAMERA_IMPL_H

#include "camera.h"

using namespace torch::indexing;

namespace mc3d
{
    Camera::Camera(std::string id, Tensor A, Tensor d, Tensor P, unsigned int height, unsigned int width, RealType distance) :
        id(id)
    {
        calibrate(A, d, P, height, width, distance);
    }

    void Camera::calibrate(Tensor A, Tensor d, Tensor P, unsigned int height, unsigned int width, RealType distance)
    {
        this->A = A;
        this->d = d;
        this->P = P;
        this->height = height;
        this->width = width;
        this->distance = distance;
        this->Ainv = A.inverse();
        this->R = P.index({Slice(0, 3), Slice(0, 3)});
        this->RT = R.transpose(0, 1);
        this->t = P.index({Slice(0, 3), Slice(3, 4)});
    }

    Point2 Camera::transform_3d_to_2d(Point3 P) const
    {
        Point3 PC = RT.mm(P - t);
        return A.mm(PC / PC[2]).index({Slice(0, 2), Slice(0, 1)});
    }

    Point3 Camera::transform_2d_to_3d(Point2 p, RealType distance) const
    {
        Point3 new_PI = torch::ones({3, 1}, TensorRealTypeOption);
        std::cout << "new_PI" << new_PI << std::endl;
        std::cout << "p" << p << std::endl;
        std::cout << "new_PI.index({Slice(0, 2), Slice(0, 1)})" << new_PI.index({Slice(0, 2), Slice(0, 1)}) << std::endl;
        new_PI.index({Slice(0, 2), Slice(0, 1)}) = p;
        return R.mm(((Ainv.mm(new_PI)) * distance)) + t;
    }
}

#endif
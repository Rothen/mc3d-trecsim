#ifndef TENSOR_FUNCTIONS_H
#define TENSOR_FUNCTIONS_H

#include "mc3d_common.h"

namespace mc3d
{
    class Project2DTo3D : public Function<Project2DTo3D>
    {
    public:
        static Tensor forward(AutogradContext *ctx, Tensor input, Tensor weight, Tensor bias = Tensor());

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
    };

    class Project3DTo2D : public Function<Project3DTo2D>
    {
    public:
        static Tensor forward(AutogradContext *ctx, Tensor x, Tensor c);

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
    };
}

#endif
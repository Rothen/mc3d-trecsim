#ifndef TENSOR_FUNCTIONS_IMPL_H
#define TENSOR_FUNCTIONS_IMPL_H

#include "tensor_functions.h"

namespace mc3d
{
    Tensor Project2DTo3D::forward(AutogradContext *ctx, Tensor input, Tensor weight, Tensor bias = Tensor())
    {
        ctx->save_for_backward({input, weight, bias});
        auto output = input.mm(weight.t());
        if (bias.defined())
        {
            output += bias.unsqueeze(0).expand_as(output);
        }
        return output;
    }

    tensor_list Project2DTo3D::backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];
        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        auto grad_bias = Tensor();
        if (bias.defined())
        {
            grad_bias = grad_output.sum(0);
        }

        return {grad_input, grad_weight, grad_bias};
    }

    // auto y = Project2DTo3D::apply(x, weight);
    // y.sum().backward();

    Tensor Project3DTo2D::forward(AutogradContext * ctx, Tensor x, Tensor c){
        // ctx is a context object that can be used to stash information
        // for backward computation
        ctx->saved_data["c"] = c;
        return x * c;
    }

    tensor_list Project3DTo2D::backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        // We return as many input gradients as there were arguments.
        // Gradients of non-tensor arguments to forward must be `Tensor()`.
        return {grad_outputs[0] * ctx->saved_data["c"].toDouble(), Tensor()};
    }

    // auto y = Project3DTo2D::apply(x, weight);
}; // namespace mc3d

#endif
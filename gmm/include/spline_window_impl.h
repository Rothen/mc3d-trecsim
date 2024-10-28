#ifndef SPLINE_WINDOW_IMPL_H
#define SPLINE_WINDOW_IMPL_H

#include "spline_window.h"

namespace mc3d
{
    SplineWindow::SplineWindow(Tensor knots, size_t degree) : degree(degree), knots(knots)
    {
        for (int i = 0; i < degree; i++)
        {
            this->knots = torch::cat({this->knots[0].reshape(1), this->knots, this->knots[-1].reshape(1)});
        }

        nb_basis = this->knots.size(0) - degree - 1;
    }

    void SplineWindow::prepare_time_projection(RealType time_delta_forward)
    {
        
    }

    // shift forward according to time_delta_forward
    SplineParameter SplineWindow::project_forward_spline_parameter(SplineParameter spline_parameter)
    {

    }

    // use design_matrix
    Point3 SplineWindow::evaluate(RealType time, SplineParameter spline_parameter)
    {
        return design_matrix(time).mm(spline_parameter).transpose(0, 1);
    }

    // val tensor, no grad
    Tensor SplineWindow::spline_smoothness_log_prior(SplineParameter spline_parameter)
    {

    }

    // according to Cox DeBoor
    inline RealType SplineWindow::basis(RealType t, int i, int k)
    {
        if (k == 0)
        {
            return RealType(knots[i].item().to<RealType>() <= t && t < knots[i + 1].item().to<RealType>());
        }

        RealType denom1 = knots[i + k].item().to<RealType>() - knots[i].item().to<RealType>();
        RealType denom2 = knots[i + k + 1].item().to<RealType>() - knots[i + 1].item().to<RealType>();

        RealType term1 = (denom1 != 0) ? (t - knots[i].item().to<RealType>()) / denom1 * basis(t, i, k - 1) : 0;
        RealType term2 = (denom2 != 0) ? (knots[i + k + 1].item().to<RealType>() - t) / denom2 * basis(t, i + 1, k - 1) : 0;

        return term1 + term2;
    }

    Tensor SplineWindow::design_matrix(RealType time_point)
    {
        Tensor r_design_matrix = torch::zeros({1, static_cast<long>(nb_basis)}, TensorRealTypeOption.requires_grad(false));

        for (int i = 0; i < nb_basis; i++)
        {
            r_design_matrix[0][i] = basis(time_point, i, degree);
        }

        return r_design_matrix;
    }

    Tensor SplineWindow::design_matrix(Tensor time_points)
    {
        Tensor r_design_matrix = torch::zeros({time_points.size(0), static_cast<long>(nb_basis)}, TensorRealTypeOption.requires_grad(false));

        for (int j = 0; j < time_points.size(0); j++)
        {
            for (int i = 0; i < nb_basis; i++)
            {
                r_design_matrix[j][i] = basis(time_points[j].item().to<RealType>(), i, degree);
            }
        }

        return r_design_matrix;
    }
}

#endif
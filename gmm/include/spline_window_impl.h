#ifndef SPLINE_WINDOW_IMPL_H
#define SPLINE_WINDOW_IMPL_H

#include "spline_window.h"

namespace mc3d
{
    SplineWindow::SplineWindow(Tensor knots, size_t degree, RealType lambda) : degree(degree), knots(knots), lambda(lambda)
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
        Tensor r_spline_smoothness_log_prior = torch::zeros({static_cast<long>(nb_basis) - 2, spline_parameter.size(0)},
            TensorRealTypeOption.requires_grad(false));

        RealType N;
        RealType s;
        RealType d_j;
        RealType b0;
        RealType b2;
        RealType b1;
        RealType r_degree = static_cast<RealType>(degree);

        for (int j = 2; j < nb_basis; j++)
        {
            /*auto basisFunc = [&](RealType t)
            {
                return basis(t, j, degree - 2);
            };*/
            N = basis_int(j, degree - 2);
            s = std::sqrt(N);
            d_j = (r_degree - 1.0) * (r_degree - 2.0) * s / (knots[j + degree - 2] - knots[j]).item().to<RealType>();
            b0 = d_j / (knots[j + degree - 2] - knots[j - 1]).item().to<RealType>();
            b2 = d_j / (knots[j + degree - 1] - knots[j]).item().to<RealType>();
            b1 = -(b0 + b2);
            r_spline_smoothness_log_prior[j - 2][j - 2] = lambda * b0;
            r_spline_smoothness_log_prior[j - 2][j - 1] = lambda * b1;
            r_spline_smoothness_log_prior[j - 2][j - 0] = lambda * b2;
            std::cout << "r_spline_smoothness_log_prior: " << r_spline_smoothness_log_prior << std::endl;
        }

        return r_spline_smoothness_log_prior;
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

    inline RealType SplineWindow::basis_int(const int j, const int k)
    {
        /*Scalar h{(knots[knots.size() - 1] - knots[0]) / 100};
        Scalar integral{0.5 * (basis(knots[0], j, k) + basis(knots[knots.size() - 1], j, k))};
        for (int i = 1; i < 100; ++i)
        {
            integral += basis(knots[0] + i * h, j, k);
        }
        integral *= h;
        return integral;*/
        return (knots[j + k + 1] - knots[j]).item().to<RealType>() / (static_cast<RealType>(k) + 1.0);
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
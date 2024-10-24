#ifndef SPLINE_WINDOW_H
#define SPLINE_WINDOW_H

#include "mc3d_common.h"

namespace mc3d
{
    using SplineParameter = Tensor; // \in R^{nb_spline_parameter x 3}

    class SplineWindow
    {
    public:
        // 3D Spline: D -> R^3
        size_t nb_spline_parameter;
        
        SplineWindow(int degree = 3);
        void prepare_time_projection(RealType time_delta_forward);
        SplineParameter project_forward_spline_parameter(SplineParameter spline_parameter); // shift forward according to time_delta_forward
        Point3 evaluate(RealType time, SplineParameter spline_parameter); // use design_matrix
        Tensor spline_smoothness_log_prior(SplineParameter spline_parameter); // val tensor, no grad
    protected:
        int degree;
        int nb_basis;

        RealType basis(RealType t, int i, int k);  // according to Cox DeBoor
        Tensor design_matrix(RealType time_point); // no grad
    };
}

#endif
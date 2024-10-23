#ifndef SPLINE_WINDOW_H
#define SPLINE_WINDOW_H

#include "mc3d_common.h"

namespace mc3d
{
    class SplineWindow
    {
    public:
        // 3D Spline: D -> R^3
        size_t nb_spline_parameter;
        using SplineParameter = Tensor; // Tensor.sizes() == {nb_spline_parameter,3}
        SplineWindow();
        void prepare_time_projection(RealType time_delta_forward);
        SplineParameter project_forward_spline_parameter(SplineParameter parameter); // shift forward according to time_delta_forward
        Point3 evaluate(RealType time, SplineParameter parameter);                   // use design_matrix
        Tensor spline_smoothness_log_prior(SplineParameter spline_parameter);        // val tensor, no grad
    protected:
        Tensor design_matrix(RealType time_point); // no grad
    };
}

#endif
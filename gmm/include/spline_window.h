#ifndef SPLINE_WINDOW_H
#define SPLINE_WINDOW_H

#include "mc3d_common.h"

namespace mc3d
{
    using SplineParameter = Tensor; // \in R^{nb_spline_parameter x 3}
    enum AugmentationMode
    {
        None,
        Same,
        Uniform
    };

    class SplineWindow
    {
    public:
        // 3D Spline: D -> R^3
        size_t nb_spline_parameter;

        SplineWindow(Tensor knots, size_t degree = 3, RealType lambda = 0.0, AugmentationMode augmentation_mode = AugmentationMode::Same);
        void prepare_time_projection(RealType time_delta_forward);
        SplineParameter project_forward_spline_parameter(SplineParameter spline_parameter); // shift forward according to time_delta_forward
        Point3 evaluate(RealType time, SplineParameter spline_parameter); // use design_matrix
        Tensor spline_smoothness_log_prior(SplineParameter spline_parameter); // val tensor, no grad
        const Tensor get_smoothing_design_matrix() const { return smoothing_design_matrix; }
        const size_t get_nb_basis() const { return nb_basis; }
    protected:
        size_t degree;
        size_t nb_basis;
        Tensor knots;
        RealType lambda;
        AugmentationMode augmentation_mode;
        Tensor smoothing_design_matrix;

        inline RealType basis(RealType t, int i, int k); // according to Cox DeBoor
        inline RealType basis_int(const int j, const int k);
        Tensor design_matrix(RealType time_point); // no grad
        Tensor design_matrix(Tensor time_points); // no grad
        void calc_smoothing_design_matrix();
    };
}

#include "spline_window_impl.h"

#endif
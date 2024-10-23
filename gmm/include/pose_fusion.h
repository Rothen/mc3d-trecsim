#ifndef POSE_ID_MEASUREMENT_H
#define POSE_ID_MEASUREMENT_H

#include "mc3d_common.h"
#include "person_hypothesis.h"
#include "spline_window.h"
#include "camera.h"
#include "pose_id_measurement.h"

namespace mc3d
{
    class Pose_Fusion
    {
    public:
        // interface to configure cameras
        // interface to init
        // interface to fill new measurements and then to prepare time window step and project_forward_spline_parameter on all hyps
        void run_em(); // use optimizer for em-step (compute_responsibilities,maximize_hypothesis) check for birth/dead
        // interface to predict/plot/... current 3D poses (all hypothesis)
    protected:
        void birth();
        void death();
        void compute_responsibilities();                        // log_sum_exp with pose_id_log_likelihood on all hypothesis,measurement pairs then detach from grad!
        void maximize_hypothesis(PersonHypothesis &hypothesis); // use slice responsibilities and pose_id_log_likelihood together with hypothesis.parameter_prior

        SplineWindow spline;
        vector<Camera> cameras;
        vector<PersonHypothesis> hypothesis;
        vector<PoseIdMesurement> measurements;
        Tensor responsibilities; // responsibilities.sizes() == {hypothesis.size(),measurements.size()}
    };
}

#endif
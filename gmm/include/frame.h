#ifndef FRAME_H
#define FRAME_H

#include "config.h"
#include "mc3d_common.h"
#include "bspline.h"
#include "camera.h"
#include "multivariate_normal.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    struct Frame
    {
        size_t cameraIndex;
        std::vector<RowMatrix<Scalar>> kpts;
        Scalar time;
        Scalar origTimestamp;

        Frame(size_t cameraIndex, std::vector<RowMatrix<Scalar>> kpts, Scalar time, Scalar origTimestamp) :
            cameraIndex(cameraIndex),
            kpts(kpts),
            time(time),
            origTimestamp(origTimestamp)
        { }
    };
}
#endif
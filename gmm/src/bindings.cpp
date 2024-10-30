#include <string>
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "mc3d_common.h"
#include "pose_parameter.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace mc3d;

PYBIND11_MODULE(gmm, m)
{
    py::class_<PoseParameter>(m, "PoseParameter")
        .def(py::init<size_t, vector<pair<size_t, size_t>>, vector<RealType>>(),
             py::arg("nb_keypoints"),
             py::arg("edges"),
             py::arg("average_limb_lengths"))
        .def_readonly("nb_keypoints", &PoseParameter::nb_keypoints)
        .def_readonly("edges", &PoseParameter::edges)
        .def_readonly("average_limb_lengths", &PoseParameter::average_limb_lengths);

    m.doc() = "MC3D-TRECSIM c++ implementation.";
}

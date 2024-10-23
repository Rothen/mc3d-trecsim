#ifndef MC3D_SHARED_H
#define MC3D_SHARED_H

#include <torch/torch.h>

using namespace torch::autograd;
using std::array;
using std::vector;
using torch::Tensor;
using torch::kDouble;

using RealType = double;
using Point3 = Tensor; // Tensor.sizes() == {3,1}
using Point2 = Tensor; // Tensor.sizes() == {2,1}

#endif
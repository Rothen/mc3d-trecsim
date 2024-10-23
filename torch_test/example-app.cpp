#include <torch/torch.h>
#include <vector>
#include <array>
#include <iostream>
#include <numbers>

using namespace torch::autograd;
using std::array;
using std::vector;
using torch::Tensor;

Tensor do_math(Tensor x)
{
    auto y = x + 2;
    auto z = y * y * 3;
    return z;
}

Tensor do_some_more_math(Tensor x,Tensor y)
{
    return y+x;
}

void some_ops()
{
    auto x = torch::ones({2, 2}, torch::requires_grad());
    Tensor y{do_math(x)};
    y = do_some_more_math(x,y);
    Tensor out{y.mean()};
    out.backward();
    std::cout << out << std::endl;
    std::cout << x.grad() << std::endl;
}

int main()
{
    some_ops();
}
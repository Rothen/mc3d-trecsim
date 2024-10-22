#include <torch/torch.h>
#include <vector>
#include <array>
#include <iostream>
#include <numbers>

using namespace torch::autograd;
using std::array;
using std::vector;
using torch::Tensor;

class Project2DTo3D : public Function<Project2DTo3D>
{
public:
    static Tensor forward(
        AutogradContext *ctx, Tensor input, Tensor weight, Tensor bias = Tensor())
    {
        ctx->save_for_backward({input, weight, bias});
        auto output = input.mm(weight.t());
        if (bias.defined())
        {
            output += bias.unsqueeze(0).expand_as(output);
        }
        return output;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];
        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        auto grad_bias = Tensor();
        if (bias.defined())
        {
            grad_bias = grad_output.sum(0);
        }

        return {grad_input, grad_weight, grad_bias};
    }
};

// auto y = Project2DTo3D::apply(x, weight);
// y.sum().backward();

class Project3DTo2D : public Function<Project3DTo2D>
{
public:
    static Tensor forward(AutogradContext *ctx, Tensor x, Tensor c)
    {
        // ctx is a context object that can be used to stash information
        // for backward computation
        ctx->saved_data["c"] = c;
        return x * c;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        // We return as many input gradients as there were arguments.
        // Gradients of non-tensor arguments to forward must be `Tensor()`.
        return {grad_outputs[0] * ctx->saved_data["c"].toDouble(), Tensor()};
    }
};

// auto y = Project3DTo2D::apply(x, weight);

/*
using RealType = double;
using Point3 = Tensor;//Tensor.sizes() == {3,1}
using Point2 = Tensor;//Tensor.sizes() == {2,1}


class Cam
{
public:
    Cam();
    void calibrate();
    Point2 transform(Point3) const;
protected:

};

class SplineWindow
{
public:
    //3D Spline: D -> R^3
    size_t nb_spline_parameter;
    using SplineParameter = Tensor; //Tensor.sizes() == {nb_spline_parameter,3}
    SplineWindow();
    void prepare_time_projection(RealType time_delta_forward);
    SplineParameter project_forward_spline_parameter(SplineParameter parameter);//shift forward according to time_delta_forward
    Point3 evaluate(RealType time,SplineParameter parameter);//use design_matrix
    Tensor spline_smoothness_log_prior(SplineParameter spline_parameter); //val tensor, no grad
protected:
    Tensor design_matrix(RealType time_point); //no grad
};

struct PoseParameter
{
    size_t nb_keypoints;
    vector<std::pair<size_t,size_t>> edges;
    vector<RealType> average_limb_lengths; //index according to edges
};

class PersonHypothesis
{
public:
    PersonHypothesis(SplineWindow& spline,const PoseParameter& pose_pars);

public:
    Tensor parameter_log_prior() const; //sum all priors
    Tensor predict(size_t limb_id, RealType time, const Cam& cam) const; //use  spline.evaluate with slice of spline_parameter and cam.transform
    void project_forward_spline_parameter(); // use spline project_forward_spline_parameter on slices
protected:
    Tensor smoothness_log_prior() const; // use spline.spline_smoothness_log_prior
    Tensor limb_length_log_prior() const; //use pose_pars.average_limb_lengths
    Tensor scale_limb_length_log_prior() const;

protected:
    Tensor spline_parameter;//spline_parameter.sizes() == {pose_pars.nb_keypoints,nb_spline_parameter,3} requires_grad
};

struct PoseMeasurement
{
    RealType time;
    const Cam& cam;
    vector<size_t> limb_ids;
    Tensor values; //assert(values.sizes() == {limb_ids.size(),2})
};

class PoseIdMesurement
{
    PoseIdMesurement(RealType prec_variance)
    :log_normalization{-std::log(prec_variance)- std::log(RealType(2)*std::numbers::pi_v<RealType>)}
    {}
    //clear,add measurement, ...

    Tensor pose_id_log_likelihood(const PersonHypothesis& hypothesis) const
    {
        using torch::indexing::Slice;
        Tensor sum{torch::zeros({1})}; //use global shared torch::TensorOptions() object to define type, grad,...
        static constexpr RealType half{RealType(1)/RealType(2)};
        for(auto& measurement : measurements)
        {
            for(size_t m_ind{0},end{measurement.limb_ids.size()}; m_ind < end; ++m_ind)
            {
                Tensor tmp{measurement.values.index({m_ind,Slice()})-hypothesis.predict(measurement.limb_ids[m_ind],measurement.time,measurement.cam)};
                sum -= half*torch::inner(tmp,tmp)+log_normalization;
            }
        }
        return sum;
    }

protected:
    RealType log_normalization;
    vector<PoseMeasurement> measurements;
};


class Pose_Fusion
{
public:
    //interface to configure cams
    //interface to init
    //interface to fill new measurements and then to prepare time window step and project_forward_spline_parameter on all hyps
    void run_em();//use optimizer for em-step (compute_responsibilities,maximize_hypothesis) check for birth/dead
    //interface to predict/plot/... current 3D poses (all hypothesis)
protected:
    void birth();
    void death();
    void compute_responsibilities(); //log_sum_exp with pose_id_log_likelihood on all hypothesis,measurement pairs then detach from grad!
    void maximize_hypothesis(PersonHypothesis& hypothesis); //use slice responsibilities and pose_id_log_likelihood together with hypothesis.parameter_prior

    SplineWindow spline;
    vector<Cam> cams;
    vector<PersonHypothesis> hypothesis;
    vector<PoseIdMesurement> measurements;
    Tensor responsibilities; //responsibilities.sizes() == {hypothesis.size(),measurements.size()}
};*/

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

void custom_ops()
{
    auto x = torch::randn({2, 3}, torch::requires_grad());
    auto weight = torch::randn({4, 3}, torch::requires_grad());
    auto y = Project2DTo3D::apply(x, weight);
    y.sum().backward();

    std::cout << x.grad() << std::endl;
    std::cout << weight.grad() << std::endl;
}

int main()
{
    some_ops();
    custom_ops();
}
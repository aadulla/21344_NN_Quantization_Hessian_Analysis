#include <torch/torch.h>
#include <math.h>
#include <signal.h>

#include "inc/utils.h"
#include "inc/hessian.h"
#include "inc/net.h"

#define MAX_ITERS 10000
#define TOL 1e-10

using namespace std;

torch::Tensor normalize(torch::Tensor &v) {
    auto factor = sqrt(torch::sum(v * v).item<float>());
    return v / factor;
}

template <typename Data_Loader>
tuple<torch::Tensor, torch::Tensor> hessian_top_eig(Net& model,
                                                    Data_Loader& data_loader,
                                                    int layer_num) {
    
    auto opts = create_tensor_opts<float>().requires_grad(true);
    torch::Tensor total_loss = torch::zeros({1}, opts);

    model.zero_grad();

    // iterate through batches and accumulate loss
    int num_batches = 0;
    for (auto& batch : data_loader) {
        num_batches++;

        // send to device
        auto data = batch.data;
        auto targets = batch.target;

        // forward-prop
        auto output = model.forward(data);
        total_loss = total_loss + nll_loss(output, targets);
    }

    total_loss /= num_batches;
    total_loss.backward(/*grad_outputs*/{},
                        /*retain_graph=*/true,
                        /*create_graph=*/true);

    // get jacobian (ensure we are only dealing with square matrices)
    auto weight = model.get_layer_weight_no_detach(layer_num);
    assert(weight.size(0) == weight.size(1));
    auto grad = weight.grad();

    // create random initial iterate
    auto v = torch::randn(weight.sizes());
    v = normalize(v);

    torch::Tensor best_eigval;
    torch::Tensor best_eigvec;
    
    // power iteration until convergence
    for (int i=0; i < MAX_ITERS; i++) {
        model.zero_grad();
        // form Hessian-vector product
        auto Hv = torch::autograd::grad({grad}, 
                                        {weight}, 
                                        /*grad_outputs=*/{v},
                                        /*retain_graph=*/true)[0];

        // eigenvalue by rayleigh quotient
        auto eigval = torch::sum(Hv.reshape({-1}) * v.reshape({-1}));
        // corresponding eigenvector
        v = normalize(Hv);

        best_eigval = eigval;
        best_eigvec = v;

        // check for convergence
        if (i!=0) {
            float delta = ((eigval - best_eigval) / best_eigval).item<float>();
            if (delta < TOL)
                break;
        }
    }

    model.zero_grad();
    return make_tuple(best_eigval.detach(), best_eigvec.detach());
}


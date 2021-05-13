#ifndef _NET_H_
#define _NET_H_

#include <torch/torch.h>
#include <vector>

#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define NUM_CLASSES 10

using namespace std;

struct Net : torch::nn::Module {

    // attributes
    vector<torch::nn::Linear> layers;

    // methods
    Net(const vector<int>& layer_sizes);
    torch::Tensor forward(torch::Tensor x);
    int get_num_layers();
    torch::Tensor get_layer_weight(int layer_num);
    torch::Tensor get_layer_weight_no_detach(int layer_num);
    void set_layer_weight(int layer_num, torch::Tensor& weight);

};

#endif /* _NET_H_ */




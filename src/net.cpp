#include <torch/torch.h>
#include <vector>

#include "inc/net.h"

using namespace std;

Net::Net(const vector<int>& layer_sizes) {
    int prev_size = MNIST_WIDTH * MNIST_HEIGHT;
    int i;
    for (i=0; i < layer_sizes.size(); i++) {
        string name = "fc" + to_string(i);
        auto opts = torch::nn::LinearOptions(prev_size, layer_sizes[i]).bias(false);
        torch::nn::Linear layer(opts);
        register_module(name, layer);
        this->layers.push_back(layer);
        prev_size = layer_sizes[i];
    }
    string name = "fc" + to_string(i);
    torch::nn::Linear layer(prev_size, NUM_CLASSES);
    register_module(name, layer);
    this->layers.push_back(layer);
}

// x: batch_size x MNIST_WIDTH x MNIST_HEIGHT
torch::Tensor Net::forward(torch::Tensor x) {
    // flatten image to vector --> x: batch_size x (MNIST_WIDTH x MNIST_HEIGHT)
    x = x.view({x.size(0), MNIST_WIDTH * MNIST_HEIGHT});
    // pass through n-1 layers (linear + relu)
    int num_layers = this->layers.size();
    for (int i=0; i < num_layers-1; i++)
        x = torch::relu(this->layers[i]->forward(x));
    // pass through last layer (linear)
    x = this->layers[num_layers-1]->forward(x);
    // softmax outputs on feature dim (1)
    return torch::log_softmax(x, 1);
}

int Net::get_num_layers() {
    return this->layers.size();
}

torch::Tensor Net::get_layer_weight(int layer_num) {
    assert(layer_num < this->layers.size());
    return this->layers[layer_num]->weight.clone().detach();
}

torch::Tensor Net::get_layer_weight_no_detach(int layer_num) {
    assert(layer_num < this->layers.size());
    return this->layers[layer_num]->weight;
}

void Net::set_layer_weight(int layer_num, torch::Tensor& weight) {
    this->layers[layer_num]->weight = weight.clone().detach().set_requires_grad(true);
    assert(this->layers[layer_num]->weight.requires_grad());
}
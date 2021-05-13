#include <vector>
#include <iostream>
#include <stdio.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "inc/net.h"
#include "inc/train_test.h"
#include "inc/generate_data.h"
#include "inc/quantizer.h"


using namespace std;
using json = nlohmann::json;

////////////////////////////////////////////////////////////////////////////////
// LAYER
////////////////////////////////////////////////////////////////////////////////
Layer_Hessian_Data::Layer_Hessian_Data(float perturb_min, 
                                       float perturb_max,
                                       int num_steps) {                 
    // setup perturb_amts
    float perturb_delta = (perturb_max - perturb_min) / num_steps;
    for (float amt=perturb_min; amt <= perturb_max; amt+=perturb_delta) {
        this->perturb_amts.push_back(amt);
    }
}

template <typename Data_Loader>
void Layer_Hessian_Data::generate_data(Net& model, 
                                  Data_Loader& data_loader, 
                                  int layer_num) {
    // get top hessian eig
    auto eigval_eigvec = hessian_top_eig(model, data_loader, layer_num);
    this->top_eigval = get<0>(eigval_eigvec);
    auto eigvec = get<1>(eigval_eigvec);

    // save original weight
    torch::Tensor orig_weight = model.get_layer_weight(layer_num);

    // iterate through perturbations
    for (float amt : this->perturb_amts) {
        // new weight = old weight + amt * eigvec
        torch::Tensor new_weight = orig_weight + (amt * eigvec);
        model.set_layer_weight(layer_num, new_weight);

        // record loss and accuracy
        auto loss_acc = test(model, data_loader);
        float loss = get<0>(loss_acc);
        float acc = get<1>(loss_acc);
        this->perturb_losses.push_back(loss);
        this->perturb_accs.push_back(acc);
    }

    // restore original weight
    model.set_layer_weight(layer_num, orig_weight);
}

json Layer_Hessian_Data::dump_json() {
    json j =    {
                    {"top_eigval", this->top_eigval.item<float>()},
                    {"perturb_amts", this->perturb_amts},
                    {"perturb_losses", this->perturb_losses},
                    {"perturb_accs", this->perturb_accs}
                };
    return j;
}

void Layer_Hessian_Data::print() {
    char buf[100];
    snprintf(buf, sizeof(buf), 
             "| %15s | %15s | %15s |", 
             "Perturb Amt", "Loss", "Accuracy");
    string header = buf;
    string divider = "";
    for (int i=0; i < header.size(); i++) 
        divider += "-";

    cout << divider << endl;
    cout << header << endl;
    cout << divider << endl;

    for (int i=0; i < this->perturb_amts.size(); i++) {
        float amt = this->perturb_amts[i];
        float loss = this->perturb_losses[i];
        float acc = this->perturb_accs[i];

        snprintf(buf, sizeof(buf), 
                 "| %15.3f | %15.3f | %15.3f |", 
                 amt, loss, acc);
        string data = buf;
        cout << data << endl;
        cout << divider << endl;
    }
}


////////////////////////////////////////////////////////////////////////////////
// BASE NET
////////////////////////////////////////////////////////////////////////////////
Net_Hessian_Data::Net_Hessian_Data(vector<int>& layer_nums,
                                   float perturb_min,
                                   float perturb_max, 
                                   int num_steps) {
    
    this->layer_nums = layer_nums;

    // setup layer hessian data objects
    for (int i : this->layer_nums) {
        Layer_Hessian_Data* layer_data = new Layer_Hessian_Data(perturb_min,
                                                                perturb_max,
                                                                num_steps);
        this->layer_datas.push_back(layer_data);
    }
}

Net_Hessian_Data::~Net_Hessian_Data() {
    for (int i=0; i < this->layer_datas.size(); i++)
        delete this->layer_datas[i];
}

// should be overriden
torch::Tensor Net_Hessian_Data::quantize_layer_weight(Net& model, int layer_num) {
    assert(false);
}

template <typename Data_Loader>
void Net_Hessian_Data::generate_data(Net& model, 
                                Data_Loader& data_loader) {
    for (int i=0; i < this->layer_datas.size(); i++) {
        int layer_num = this->layer_nums[i];
        // save original weight
        torch::Tensor orig_weight = model.get_layer_weight(layer_num);
        // get new weight (quantize->dequantize)
        auto q_weight = this->quantize_layer_weight(model, layer_num);
        model.set_layer_weight(layer_num, q_weight);
        // generate data with new weight
        this->layer_datas[i]->generate_data(model, data_loader, layer_num);
        // restore original weight
        model.set_layer_weight(layer_num, orig_weight);
    }
}

json Net_Hessian_Data::dump_json() {
    json net_j;
    for (int i=0; i < this->layer_datas.size(); i++) {
        int layer_num = this->layer_nums[i];
        json layer_j = this->layer_datas[i]->dump_json();
        string layer_header = "layer_" + to_string(layer_num);
        net_j[layer_header] = layer_j;
    }
    return net_j;
}

void Net_Hessian_Data::print() {
    for (int i=0; i < this->layer_datas.size(); i++) {
        auto layer_data = this->layer_datas[i];
        cout << "Layer " << this->layer_nums[i] << endl;
        cout << "=====================" << endl;
        layer_data->print();
        cout << endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// NO Q NET
////////////////////////////////////////////////////////////////////////////////
No_Q_Net_Hessian_Data::No_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                             float perturb_min,
                                             float perturb_max, 
                                             int num_steps) :
Net_Hessian_Data(layer_nums, perturb_min, perturb_max, num_steps) {}

torch::Tensor No_Q_Net_Hessian_Data::quantize_layer_weight(Net& model, int layer_num) {
    return model.get_layer_weight(layer_num);
}

////////////////////////////////////////////////////////////////////////////////
// AFFINE Q NET
////////////////////////////////////////////////////////////////////////////////
Affine_Q_Net_Hessian_Data::Affine_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                                     float perturb_min,
                                                     float perturb_max, 
                                                     int num_steps) :
Net_Hessian_Data(layer_nums, perturb_min, perturb_max, num_steps) {}

torch::Tensor Affine_Q_Net_Hessian_Data::quantize_layer_weight(Net& model, int layer_num) {
    torch::Tensor weight = model.get_layer_weight(layer_num);
    Eigen::MatrixXf E = torch_to_eigen<float>(weight);
    auto Q = affine_quantize<float, int8_t>(E);
    Eigen::MatrixXf DQ = Q.dequantize();
    return eigen_to_torch<float>(DQ).set_requires_grad(true);
}

////////////////////////////////////////////////////////////////////////////////
// SCALE Q NET
////////////////////////////////////////////////////////////////////////////////
Scale_Q_Net_Hessian_Data::Scale_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                                   float perturb_min,
                                                   float perturb_max, 
                                                   int num_steps) :
Net_Hessian_Data(layer_nums, perturb_min, perturb_max, num_steps) {}

torch::Tensor Scale_Q_Net_Hessian_Data::quantize_layer_weight(Net& model, int layer_num) {
    torch::Tensor weight = model.get_layer_weight(layer_num);
    Eigen::MatrixXf E = torch_to_eigen<float>(weight);
    auto Q = scale_quantize<float, int8_t>(E);
    Eigen::MatrixXf DQ = Q.dequantize();
    return eigen_to_torch<float>(DQ).set_requires_grad(true);
}

////////////////////////////////////////////////////////////////////////////////
// KL DIV Q NET
////////////////////////////////////////////////////////////////////////////////
KL_Div_Q_Net_Hessian_Data::KL_Div_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                                     float perturb_min,
                                                     float perturb_max, 
                                                     int num_steps) :
Net_Hessian_Data(layer_nums, perturb_min, perturb_max, num_steps) {}

torch::Tensor KL_Div_Q_Net_Hessian_Data::quantize_layer_weight(Net& model, int layer_num) {
    torch::Tensor weight = model.get_layer_weight(layer_num);
    Eigen::MatrixXf E = torch_to_eigen<float>(weight);
    auto Q = kl_div_quantize<float, int8_t>(E, this->num_raw_bins);
    Eigen::MatrixXf DQ = Q.dequantize();
    return eigen_to_torch<float>(DQ).set_requires_grad(true);
}
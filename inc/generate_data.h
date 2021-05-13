#ifndef _GENERATE_DATA_H_
#define _GENERATE_DATA_H_

#include <vector>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "inc/net.h"

using namespace std;
using json = nlohmann::json;

////////////////////////////////////////////////////////////////////////////////
// LAYER
////////////////////////////////////////////////////////////////////////////////
class Layer_Hessian_Data {

    private:
        torch::Tensor top_eigval;
        vector<float> perturb_amts;
        vector<float> perturb_accs;
        vector<float> perturb_losses;

    public:
        Layer_Hessian_Data(float perturb_min, float perturb_max, int num_steps);
        template <typename Data_Loader>
        void generate_data(Net& model, Data_Loader& data_loader, int layer_num);
        json dump_json();
        void print();
};

////////////////////////////////////////////////////////////////////////////////
// BASE NET
////////////////////////////////////////////////////////////////////////////////
class Net_Hessian_Data {

    private:
        vector<int> layer_nums;
        vector<Layer_Hessian_Data*> layer_datas;

    public:
        Net_Hessian_Data(vector<int>& layer_nums,
                         float perturb_min,
                         float perturb_max, 
                         int num_steps);
        ~Net_Hessian_Data();
        virtual torch::Tensor quantize_layer_weight(Net& model, int layer_num);
        template <typename Data_Loader>
        void generate_data(Net& model, Data_Loader& data_loader);
        json dump_json();
        void print();
};

////////////////////////////////////////////////////////////////////////////////
// NO Q NET
////////////////////////////////////////////////////////////////////////////////
class No_Q_Net_Hessian_Data : public Net_Hessian_Data {

    public:
        No_Q_Net_Hessian_Data(vector<int>& layer_nums,
                              float perturb_min,
                              float perturb_max, 
                              int num_steps);
        torch::Tensor quantize_layer_weight(Net& model, int layer_num) override;
};

////////////////////////////////////////////////////////////////////////////////
// AFFINE Q NET
////////////////////////////////////////////////////////////////////////////////
class Affine_Q_Net_Hessian_Data : public Net_Hessian_Data {

    public:
        Affine_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                  float perturb_min,
                                  float perturb_max, 
                                  int num_steps);
        torch::Tensor quantize_layer_weight(Net& model, int layer_num) override;
};

////////////////////////////////////////////////////////////////////////////////
// SCALE Q NET
////////////////////////////////////////////////////////////////////////////////
class Scale_Q_Net_Hessian_Data : public Net_Hessian_Data {

    public:
        Scale_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                 float perturb_min,
                                 float perturb_max, 
                                 int num_steps);
        torch::Tensor quantize_layer_weight(Net& model, int layer_num) override;
};

////////////////////////////////////////////////////////////////////////////////
// KL Q NET
////////////////////////////////////////////////////////////////////////////////
class KL_Div_Q_Net_Hessian_Data : public Net_Hessian_Data {

    static const int num_raw_bins = 256;
    
    public:
        KL_Div_Q_Net_Hessian_Data(vector<int>& layer_nums,
                                  float perturb_min,
                                  float perturb_max, 
                                  int num_steps);
        torch::Tensor quantize_layer_weight(Net& model, int layer_num) override;
};

#include "src/generate_data.tpp"

#endif /* _GENERATE_DATA_H_ */
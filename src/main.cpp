#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <math.h>
#include <stdio.h>
#include <torch/torch.h>

#include "inc/net.h"
#include "inc/hessian.h"
#include "inc/globals.h"
#include "inc/train_test.h"
#include "inc/generate_data.h"

using namespace std;
using json = nlohmann::json;

void setup_torch(int seed) {
    torch::manual_seed(seed);
}

int main(int argc, char* argv[]) {
    // parse cli args
    string config_json_path(argv[1]);
    string results_json_path(argv[2]);
    // string config_json_path = "/code/exps/64x64x64/config.json";
    // string results_json_path = "./out.json";

    // read in_json
    ifstream config_json(config_json_path);
    json config = json::parse(config_json);
    cout << "Parsed " << config_json_path << endl << endl;

    // extract config parameters
    int seed =                  config["seed"].get<int>();
    vector<int> layer_sizes =   config["layer_sizes"].get<vector<int>>();
    string data_dir =           config["data_dir"].get<string>();
    int train_batch_size =      config["train_batch_size"].get<int>();
    int test_batch_size =       config["test_batch_size"].get<int>();
    int epochs =                config["epochs"].get<int>();
    vector<string> q_schemes =  config["q_schemes"].get<vector<string>>();
    vector<int> q_layer_nums =  config["q_layer_nums"].get<vector<int>>();
    float perturb_min =         config["perturb_min"].get<float>();
    float perturb_max =         config["perturb_max"].get<float>();
    int perturb_steps =         config["perturb_steps"].get<int>();

    // setup torch
    setup_torch(seed);

    // create model
    Net model = Net(layer_sizes);
    cout << model << endl << endl;

    // create train data loader
    auto train_dataset = torch::data::datasets::MNIST(data_dir).map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(move(train_dataset), train_batch_size);

    // create test data loader
    auto test_dataset = torch::data::datasets::MNIST(data_dir).map(torch::data::transforms::Stack<>());
    auto test_data_loader = torch::data::make_data_loader(move(test_dataset), test_batch_size);

    // create optimizer
    auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    // store results from training/testing and quantization
    json results;

    vector<float> train_losses;
    vector<float> train_accs;
    vector<float> test_losses;
    vector<float> test_accs;

    for (size_t epoch=1; epoch <= epochs; epoch++) {
        auto train_loss_acc = train(model, *train_data_loader, optimizer);
        float train_loss = get<0>(train_loss_acc);
        float train_acc = get<1>(train_loss_acc);
        train_losses.push_back(train_loss);
        train_accs.push_back(train_acc);

        auto test_loss_acc = test(model, *test_data_loader);
        float test_loss = get<0>(test_loss_acc);
        float test_acc = get<1>(test_loss_acc);
        test_losses.push_back(test_loss);
        test_accs.push_back(test_acc);

        cout << epoch << train_loss << train_acc << test_loss << test_acc << endl;
    }    
    cout << endl;

    // update results
    results["overall"]["train_losses"]  = train_losses;
    results["overall"]["train_accs"]    = train_accs;
    results["overall"]["test_losses"]   = test_losses;
    results["overall"]["test_accs"]     = test_accs;

    // perform different quantization schemes
    for (string q_scheme : q_schemes) {

        if (q_scheme == "no") {
            cout << "Generting Data for No Quantization..." << endl;
            No_Q_Net_Hessian_Data nhd(q_layer_nums, 
                                    perturb_min, 
                                    perturb_max, 
                                    perturb_steps);
            nhd.generate_data(model, *test_data_loader);
            results["q_schemes"][q_scheme] = nhd.dump_json();
            cout << "Data Generated!!!" << endl << endl;

        }

        else if (q_scheme == "affine") {
            cout << "Generting Data for Affine Quantization..." << endl;
            Affine_Q_Net_Hessian_Data ahd(q_layer_nums, 
                                    perturb_min, 
                                    perturb_max, 
                                    perturb_steps);
            ahd.generate_data(model, *test_data_loader);
            results["q_schemes"][q_scheme] = ahd.dump_json();
            cout << "Data Generated!!!" << endl << endl;
        }

        else if (q_scheme == "scale") {
            cout << "Generting Data for Scale Quantization..." << endl;
            Scale_Q_Net_Hessian_Data shd(q_layer_nums, 
                                    perturb_min, 
                                    perturb_max, 
                                    perturb_steps);
            shd.generate_data(model, *test_data_loader);
            results["q_schemes"][q_scheme] = shd.dump_json();
            cout << "Data Generated!!!" << endl << endl;
        }

        else if (q_scheme == "kl_div") {
            cout << "Generting Data for KL Div Quantization..." << endl;
            KL_Div_Q_Net_Hessian_Data kldhd(q_layer_nums, 
                                    perturb_min, 
                                    perturb_max, 
                                    perturb_steps);
            kldhd.generate_data(model, *test_data_loader);
            results["q_schemes"][q_scheme] = kldhd.dump_json();
            cout << "Data Generated!!!" << endl << endl;
        }
    }

    // log results
    ofstream results_json(results_json_path);
    results_json << results.dump(4);
    cout << "Logged " << results_json_path << endl << endl;

    return 0;
}

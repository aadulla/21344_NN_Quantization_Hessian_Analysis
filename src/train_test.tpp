#include <torch/torch.h>
#include <cstdio>
#include <iostream>

#include "inc/utils.h"
#include "inc/globals.h"
#include "inc/net.h"
#include "inc/train_test.h"

using namespace std;
using namespace torch;

extern Device torch_device;

////////////////////////////////////////////////////////////////////////////////
// TRAIN
////////////////////////////////////////////////////////////////////////////////
template <typename Data_Loader>
tuple<float, float> train(Net& model,
                          Data_Loader& data_loader,
                          optim::Optimizer& optimizer) {

    model.to(torch_device);
    model.train();
    float train_loss = 0.0;
    float train_acc = 0.0;
    int num_batches = 0;

    for (auto& batch : data_loader) {
        num_batches++;

        // send to device
        auto data = batch.data.to(torch_device);
        auto targets = batch.target.to(torch_device);

        // forward-prop, calculate loss, back-prop
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        // accumulate train loss
        train_loss += loss.template item<float>();

        // accumulate train accuracy
        auto pred = output.argmax(1);
        int correct = pred.eq(targets).sum().template item<int>();
        train_acc += (static_cast<float>(correct)) / (static_cast<float>(data.size(0)));
    }

    train_loss /= num_batches;
    train_acc /= num_batches;
    return make_tuple(train_loss, train_acc);
}


////////////////////////////////////////////////////////////////////////////////
// TEST
////////////////////////////////////////////////////////////////////////////////
template <typename Data_Loader>
tuple<float, float> test(Net& model,
                         Data_Loader& data_loader) {

    NoGradGuard no_grad;
    model.to(torch_device);
    model.eval();
    float test_loss = 0.0;
    float test_acc = 0.0;
    int num_batches = 0;

    for (const auto& batch : data_loader) {
        num_batches++;

        // send to device
        auto data = batch.data.to(torch_device);
        auto targets = batch.target.to(torch_device);

        // forward-prop
        auto output = model.forward(data);
        auto loss = nll_loss(output, targets);

        // accumulate loss
        test_loss += loss.template item<float>();

        // accumulate accuracy
        auto pred = output.argmax(1);
        int correct = pred.eq(targets).sum().template item<int>();
        test_acc += (static_cast<float>(correct)) / (static_cast<float>(data.size(0)));
    }

    test_loss /= num_batches;
    test_acc /= num_batches;
    return make_tuple(test_loss, test_acc);

}
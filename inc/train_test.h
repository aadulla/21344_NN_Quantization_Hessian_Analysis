#ifndef _TRAIN_TEST_H_
#define _TRAIN_TEST_H_

#include <torch/torch.h>

#include "inc/net.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// TRAIN
////////////////////////////////////////////////////////////////////////////////
template <typename Data_Loader>
tuple<float, float> train(Net& model,
                               Data_Loader& data_loader,
                               torch::optim::Optimizer& optimizer);

////////////////////////////////////////////////////////////////////////////////
// TEST
////////////////////////////////////////////////////////////////////////////////
template <typename Data_Loader>
tuple<float, float> test(Net& model,
                              Data_Loader& data_loader);

#include "src/train_test.tpp"

#endif /* _TRAIN_TEST_H_ */
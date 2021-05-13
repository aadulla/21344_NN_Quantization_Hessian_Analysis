#ifndef _HESSIAN_H_
#define _HESSIAN_H_

#include <torch/torch.h>

#include "inc/net.h"

using namespace std;

template <typename Data_Loader>
tuple<torch::Tensor, torch::Tensor> hessian_top_eig(Net& model,
                                                    Data_Loader& data_loader,
                                                    int layer_num);

#include "src/hessian.tpp"

#endif /* _HESSIAN_H_ */

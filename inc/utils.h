#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <Eigen/Dense>
#include <torch/torch.h>

#define CLAMP(low, high, x) min(high, max(low, x))

using namespace std;

// Column-Major Ordered Eigen Matrix (default)
template <typename DTYPE>
using MatrixX = typename Eigen::Matrix<DTYPE, 
                                       Eigen::Dynamic, 
                                       Eigen::Dynamic>;

// Row-Major Ordered Eigen Matrix
template <typename DTYPE>
using MatrixX_rm = typename Eigen::Matrix<DTYPE, 
                                          Eigen::Dynamic, 
                                          Eigen::Dynamic, 
                                          Eigen::RowMajor>;

template <typename INT_DTYPE> 
INT_DTYPE get_int_min() {
    return -1 * (1 << ((sizeof(INT_DTYPE) * 8) - 1));
}

template <typename INT_DTYPE> 
INT_DTYPE get_int_max() {
    return (1 << ((sizeof(INT_DTYPE) * 8) - 1)) - 1;
}

template <typename INT_DTYPE> 
uint32_t get_int_steps() {
    return (1 << (sizeof(INT_DTYPE) * 8)) - 1;
}

template <typename IN_DTYPE, typename OUT_DTYPE>
MatrixX<OUT_DTYPE> cast_eigen(const MatrixX<IN_DTYPE>& in_E) {
    int num_rows = in_E.rows();
    int num_cols = in_E.cols();
    auto out_E = MatrixX<OUT_DTYPE>(num_rows, num_cols);

    for (int i=0; i < num_rows; i++)
        for (int j=0; j < num_cols; j++)
            out_E(i,j) = (OUT_DTYPE)in_E(i,j);

    return out_E;
}

template <typename DTYPE>
inline auto create_tensor_opts() -> decltype(torch::TensorOptions()) {
    return torch::TensorOptions();
}

template < >
inline auto create_tensor_opts<float>() -> decltype(torch::TensorOptions()) {
    return torch::TensorOptions().dtype(torch::kFloat32);
}

template < >
inline auto create_tensor_opts<double>() -> decltype(torch::TensorOptions()) {
    return torch::TensorOptions().dtype(torch::kFloat64);
}

template <typename DTYPE>
torch::Tensor eigen_to_torch(const MatrixX<DTYPE>& E) {
    MatrixX_rm<DTYPE> E_rm(E); // convert E to row-major E_rm
    
    auto opts = create_tensor_opts<DTYPE>();
    auto T = torch::from_blob(E_rm.data(), {E_rm.rows(), E_rm.cols()}, opts);
    return T.clone();
}

template <typename DTYPE>
MatrixX<DTYPE> torch_to_eigen(const torch::Tensor& T) {
    torch::Tensor T_clone = T.clone();
    DTYPE* data_ptr = (DTYPE*)(T_clone.data_ptr<DTYPE>());

    Eigen::Map<MatrixX_rm<DTYPE>> E_rm(data_ptr, T.size(0), T.size(1));
    MatrixX<DTYPE> E(E_rm); // convert E_rm to column-major E
    return E;
}

#endif /* _UTILS_H_ */
#ifndef _Q_MATRIX_TPP_
#define _Q_MATRIX_TPP_

#include <math.h> 
#include <stdio.h>
#include <Eigen/Dense>

#include "inc/utils.h"
#include "inc/q_matrix.h"

using namespace std;

template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE>::Q_Matrix(MatrixX<RAW_DTYPE>& E, 
                                       RAW_DTYPE scale, 
                                       Q_DTYPE zero_point) {
    this->scale = scale;
    this->zero_point = zero_point;
    int num_rows = E.rows();
    int num_cols = E.cols();
    this->q_matrix = MatrixX<Q_DTYPE>(num_rows, num_cols);

    int q_min = get_int_min<Q_DTYPE>();
    int q_max = get_int_max<Q_DTYPE>();
    for (int i=0; i < num_rows; i++) {
        for (int j=0; j < num_cols; j++) {
            RAW_DTYPE real_val = E(i,j);
            int q_val = nearbyint(this->zero_point + (real_val / this->scale));
            this->q_matrix(i,j) = CLAMP(q_min, q_max, q_val);
        }
    }
}

template <typename RAW_DTYPE, typename Q_DTYPE>
MatrixX<RAW_DTYPE> Q_Matrix<RAW_DTYPE, Q_DTYPE>::dequantize() {
    int num_rows = this->q_matrix.rows();
    int num_cols = this->q_matrix.cols();
    auto dq_matrix = MatrixX<RAW_DTYPE>(num_rows, num_cols);

    for (int i=0; i < num_rows; i++) {
        for (int j=0; j < num_cols; j++) {
            Q_DTYPE q_val = this->q_matrix(i,j);
            RAW_DTYPE real_val = this->scale * (q_val - this->zero_point);
            dq_matrix(i,j) = real_val;
        }
    }

    return dq_matrix;
}

template <typename RAW_DTYPE, typename Q_DTYPE>
void Q_Matrix<RAW_DTYPE, Q_DTYPE>::print() {
    cout << "Scale: " << this->scale << endl;
    cout << "Zero Point: " << (int)this->zero_point << endl;

    // // cout can't print int8_t, so cast to int
    // auto int_q_matrix = cast_eigen<Q_DTYPE, int>(this->q_matrix);
    // cout << "Quantized Matrix: " << endl << int_q_matrix << endl;
}

#endif /* _Q_MATRIX_TPP_ */
#ifndef _Q_MATRIX_H_
#define _Q_MATRIX_H_

#include "inc/utils.h"

template <typename RAW_DTYPE, typename Q_DTYPE>
class Q_Matrix {
    
    private:
        RAW_DTYPE scale;
        Q_DTYPE zero_point;
        MatrixX<Q_DTYPE> q_matrix;
        
    public:
        Q_Matrix(MatrixX<RAW_DTYPE>& E, RAW_DTYPE scale, Q_DTYPE zero_point);
        MatrixX<RAW_DTYPE> dequantize();
        void print();
};

#include "src/q_matrix.tpp"

#endif /* _Q_MATRIX_H_ */
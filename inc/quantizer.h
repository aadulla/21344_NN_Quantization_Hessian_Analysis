#ifndef _QUANTIZER_H_
#define _QUANTIZER_H_

#include "inc/utils.h"
#include "inc/q_matrix.h"

template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> scale_quantize(MatrixX<RAW_DTYPE>& E);

template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> affine_quantize(MatrixX<RAW_DTYPE>& E);

template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> kl_div_quantize(MatrixX<RAW_DTYPE>& E, 
                                             size_t num_raw_bins);

#include "src/quantizer.tpp"

#endif /* _QUANTIZER_H_ */
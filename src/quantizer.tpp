#ifndef _QUANTIZER_TPP_
#define _QUANTIZER_TPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <csignal>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

#include "inc/utils.h"
#include "inc/q_matrix.h"
#include "inc/quantizer.h"

#define EPSILON 1.0e-5

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Scale Quantization
////////////////////////////////////////////////////////////////////////////////
template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> scale_quantize(MatrixX<RAW_DTYPE>& E) {
    RAW_DTYPE real_max = E.maxCoeff();
    RAW_DTYPE real_min = E.minCoeff();
    RAW_DTYPE abs_max = max(abs(real_max), abs(real_min));

    uint32_t num_levels = get_int_steps<Q_DTYPE>();

    RAW_DTYPE scale = (2 * abs_max) / num_levels;
    Q_DTYPE zero_point = 0;

    return Q_Matrix<RAW_DTYPE, Q_DTYPE>(E, scale, zero_point);
}

////////////////////////////////////////////////////////////////////////////////
// Affine Quantization
////////////////////////////////////////////////////////////////////////////////
template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> affine_quantize(MatrixX<RAW_DTYPE>& E) {
    RAW_DTYPE real_max = E.maxCoeff();
    RAW_DTYPE real_min = E.minCoeff();

    Q_DTYPE q_min = get_int_min<Q_DTYPE>();

    uint32_t num_levels = get_int_steps<Q_DTYPE>();

    RAW_DTYPE scale = (real_max - real_min) / num_levels;
    Q_DTYPE zero_point = q_min - (real_min / scale);

    return Q_Matrix<RAW_DTYPE, Q_DTYPE>(E, scale, zero_point);
}

////////////////////////////////////////////////////////////////////////////////
// KL Divergence Quantization
////////////////////////////////////////////////////////////////////////////////
typedef struct histogram {
    vector<int> counts;
    vector<double> bins;
} histogram_t;

template <typename RAW_DTYPE>
histogram_t create_histogram(const MatrixX<RAW_DTYPE>& E, size_t num_bins) {
    // create list of all values in E
    vector<double> vals;
    for (int i=0; i < E.rows(); i++)
        for (int j=0; j < E.cols(); j++)
            vals.push_back(E(i,j));

    // sort in ascending order
    sort(vals.begin(), vals.end());

    // setup bin parameters
    double min_val = vals.front();
    double max_val = vals.back();
    double bin_width = (max_val - min_val) / num_bins;
    double bin_low = min_val;
    double bin_high = bin_low + bin_width;
    int bin_count = 0;

    // initialize histogram
    histogram_t hist = {};
    hist.bins.push_back(bin_low);

    // iterate through each val and add it to histogram
    size_t idx = 0;
    while (idx < vals.size()) {
        // if this is the last bin, hardcode bin_high = max_val
        bin_high = (hist.bins.size() == num_bins) ? max_val : bin_high;
        if (vals.at(idx) <= bin_high) {
            bin_count++;
            idx++;
        }
        else {
            hist.counts.push_back(bin_count);
            hist.bins.push_back(bin_high);
            bin_count = 0;
            bin_low = bin_high;
            bin_high = bin_low + bin_width;
        }
    }

    // add any remaining vals to histogram
    hist.counts.push_back(bin_count);
    assert(hist.counts.size() == (size_t)num_bins);
    hist.bins.push_back(bin_high);
    assert(hist.bins.size() == (size_t)num_bins + 1);

    return hist;
}

double calc_kl_div(const vector<double>& P, const vector<double>& Q) {
    // validate distributions
    assert(P.size() == Q.size());
    assert(abs(accumulate(P.begin(), P.end(), 0.0) - 1) < EPSILON);
    assert(abs(accumulate(Q.begin(), Q.end(), 0.0) - 1) < EPSILON);

    double kl_div = 0;
    for (uint32_t i=0; i < P.size(); i++)
        // check for division by 0
        if (Q[i] == 0)
            kl_div += log(P[i] / EPSILON);
        else
            kl_div += log(P[i] / Q[i]);

    return kl_div;
}

vector<double> quantize_dist(const vector<double>& raw_dist, int num_q_bins) {
    // how many raw bins fit into a quantized bin
    int num_raw_bins_per_q_bin = raw_dist.size() / num_q_bins;
    // how many quantized bins have to include 1 more raw bin
    int num_overflow_q_bins = raw_dist.size() % num_q_bins;

    vector<double> q_dist;

    int raw_idx_low = 0;
    int raw_idx_high = 0;
    for (int i=0; i < num_q_bins; i++) {
        // determine raw bins interval
        raw_idx_low = raw_idx_high;
        raw_idx_high = raw_idx_low + \
                       num_raw_bins_per_q_bin + \
                       (i < num_overflow_q_bins);

        // accumulate the prob mass over the raw bins into a quantized mass
        double q_mass = accumulate(raw_dist.begin() + raw_idx_low, 
                                   raw_dist.begin() + raw_idx_high,
                                   0.0);
        
        int num_raw_bins_in_interval = raw_idx_high - raw_idx_low;

        // count number of non-zero bins in the interval
        int num_nonzero_bins_in_interval = 0;
        for (int j=0; j < num_raw_bins_in_interval; j++) 
            if (raw_dist[j] != 0)
                num_nonzero_bins_in_interval++;

        // expand the quantized mass over the quantized bins
        // puts 0s if raw bin was also 0
        for (int j=0; j < num_raw_bins_in_interval; j++)
            if (raw_dist[j] != 0)
                q_dist.push_back(q_mass / num_nonzero_bins_in_interval);
            else
                q_dist.push_back(0);

    }

    return q_dist;
}

template <typename RAW_DTYPE, typename Q_DTYPE>
Q_Matrix<RAW_DTYPE, Q_DTYPE> kl_div_quantize(MatrixX<RAW_DTYPE>& E, size_t num_raw_bins) {
    // create histogram from E
    int total_count = E.rows() * E.cols();
    histogram_t raw_hist = create_histogram(E, num_raw_bins);

    // initialize min_kl_* with sentinel values
    double min_kl_div = numeric_limits<double>::infinity();
    double min_kl_div_bin_low = raw_hist.bins.front();
    double min_kl_div_bin_high = raw_hist.bins.back();

    // number of quantized bins
    size_t num_q_bins = get_int_steps<Q_DTYPE>();

    // number of vals below the low threshold
    int low_outlier_count = 0;

    for (size_t i=0; i < num_raw_bins-num_q_bins; i++) {
        double bin_low = raw_hist.bins[i];
        // cut off all bins below i-1 so add to low_outlier_count
        if (i > 0)
            low_outlier_count += raw_hist.counts[i-1];
        
        // number of vals above the high threshold
        int high_outlier_count = accumulate(raw_hist.counts.begin() + num_q_bins + i,
                                            raw_hist.counts.end(),
                                            0);
        for (size_t j=i+num_q_bins; j < num_raw_bins; j++) {
            double bin_high = raw_hist.bins[j];
            // just included another bin j-1, so remove it from high_outlier_count
            if (j > i + num_q_bins)
                high_outlier_count -= raw_hist.counts[j-1];

            // create raw distribution from bins[i:j]
            auto raw_dist = vector<double>(raw_hist.counts.begin() + i,
                                           raw_hist.counts.begin() + j);
            raw_dist[0] += low_outlier_count;
            raw_dist[j-i-1] += high_outlier_count;
            assert(accumulate(raw_dist.begin(), raw_dist.end(), 0) == total_count);
            assert(accumulate(raw_hist.counts.begin(), raw_hist.counts.end(), 0) == total_count);

            // normalize distribution
            for (double& val: raw_dist)
                val /= total_count;

            // create quantized distribution
            auto q_dist = quantize_dist(raw_dist, num_q_bins);

            // determine KL Divergence
            double kl_div = calc_kl_div(raw_dist, q_dist);
            if (kl_div < min_kl_div) {
                min_kl_div = kl_div;
                min_kl_div_bin_low = bin_low;
                min_kl_div_bin_high = bin_high;
            }
        }
    }

    Q_DTYPE q_min = get_int_min<Q_DTYPE>();
    RAW_DTYPE scale = (min_kl_div_bin_high - min_kl_div_bin_low) / num_q_bins;
    Q_DTYPE zero_point = q_min - (min_kl_div_bin_low / scale);

    cout << min_kl_div_bin_low << " " << min_kl_div_bin_high << " " << min_kl_div << endl;

    return Q_Matrix<RAW_DTYPE, Q_DTYPE>(E, scale, zero_point);
}

#endif /* _QUANTIZER_TPP_ */
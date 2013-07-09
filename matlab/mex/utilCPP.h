#ifndef utilCPP_H_
#define utilCPP_H_

#include "mex.h"
#include "opencv2/core/core.hpp"
#include "MLData.hpp"
#include <vector>

typedef std::vector<double> VecDbl;
typedef std::vector<int> VecInt;

void set_cvdata (const mxArray *parr, cv::Mat& mat);
void set_X (const mxArray *from, cv::Mat& to);
void set_Y (const mxArray *from, cv::Mat& to);
void set_Ymc (const mxArray *from, cv::Mat& to); // multi class Y
void set_mask (const mxArray *from, std::vector<VAR_TYPE>& to);

mxArray* VecDbl_to_mxArray(const VecDbl& from);
mxArray* VecInt_to_mxArray(const VecInt& from);
mxArray* cvMatDbl_to_mxArray(const cv::Mat_<double>& from);
mxArray* cvMatInt_to_mxArray(const cv::Mat_<int>& from);

#endif
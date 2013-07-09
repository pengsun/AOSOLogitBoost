#include "utilCPP.h"


void set_X (const mxArray *from, cv::Mat& to)
{
    int nvar = mxGetM(from);
    int nsample = mxGetN(from);
    float* pdata = (float*)mxGetData(from);

    cv::Mat tmp(nsample,nvar,CV_32FC1,(uchar*)pdata);
    to = tmp;
}

void set_Y (const mxArray *from, cv::Mat& to)
{
  int nsample =  mxGetM(from) * mxGetN(from);
  uchar* pdata = (uchar*)mxGetData(from);

  cv::Mat tmp(nsample,1,CV_32FC1,(uchar*)pdata);
  to = tmp;
}

void set_Ymc( const mxArray *from, cv::Mat& to )
{
  int nsample = mxGetN(from);
  int K = mxGetM(from);
  uchar* pdata = (uchar*)mxGetData(from);

  cv::Mat tmp(nsample,K,CV_32FC1,(uchar*)pdata);
  to = tmp;
}

void set_mask( const mxArray *from, std::vector<VAR_TYPE>& to )
{
  int n = mxGetM(from) * mxGetN(from);
  to.resize(n);
  
  uchar* p = (uchar*)mxGetData(from);
  // TODO: assert mxArray is UINT8
  for (int i = 0; i < n; ++i) {
    uchar elem = *(p+i);
    to[i] = (elem!=0)? VAR_CAT : VAR_NUM;
  }
}

mxArray* VecDbl_to_mxArray( const VecDbl& from)
{
  int n = from.size();
  mxArray* to = mxCreateDoubleMatrix(n,1,mxREAL);
  double* ptr = (double*)mxGetData(to);

  std::copy(from.begin(),from.end(), ptr);

  return to;
}

mxArray* VecInt_to_mxArray( const VecInt& from)
{
  int n = from.size();
  mxArray* to = mxCreateDoubleMatrix(n,1,mxREAL);
  double* ptr = (double*)mxGetData(to);

  std::copy(from.begin(),from.end(), ptr);

  return to;
}

mxArray* cvMatDbl_to_mxArray( const cv::Mat_<double>& from )
{
  int M = from.rows;
  int N = from.cols;

  mxArray* to = mxCreateDoubleMatrix(N,M,mxREAL);
  double* ptr = (double*)mxGetData(to);

  std::copy(from.begin(),from.end(), ptr);

  return to;
}

mxArray* cvMatInt_to_mxArray( const cv::Mat_<int>& from )
{
  int M = from.rows;
  int N = from.cols;

  mxArray* to = mxCreateNumericMatrix(N, M, mxINT32_CLASS, mxREAL);
  int* ptr = (int*)mxGetData(to);

  std::copy(from.begin(),from.end(), ptr);

  return to;
}
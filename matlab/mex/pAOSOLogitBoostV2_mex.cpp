#include "mex.h"
#include "./../../src/MLData.hpp"
#include "./../../src/pAOSOLogitBoostV2.hpp"
#include "utilCPP.h"
#include <cstring>

// typedef for booster
typedef pAOSOLogitBoostV2 booster_t;

namespace {
  
}

// h = train(dummy, X,Y, var_cat_mask, T,J,v, node_size, rs,rf, wrs);
void train(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  MLData tr;
  // training tr X
  set_X(prhs[1], tr.X);
  // response Y
  set_Y(prhs[2], tr.Y);
  // var_cat_mask
  set_mask(prhs[3], tr.var_type);
  // T
  int T = (int)mxGetScalar(prhs[4]);
  // J
  int J = (int)mxGetScalar(prhs[5]);
  // v
  double v = (double)mxGetScalar(prhs[6]);
  // node_size
  int node_size = (int)mxGetScalar(prhs[7]);
  // ratio sample
  double ratio_si = (double)mxGetScalar(prhs[8]);
  // ratio feature
  double ratio_fi = (double)mxGetScalar(prhs[9]);

  // weight ratio sample
  double weight_ratio_si = (double)mxGetScalar(prhs[10]);

  /* train */
  tr.problem_type = PROBLEM_CLS;
  tr.preprocess();

  booster_t* pbooster = new booster_t;
  pbooster->param_.T = T;
  pbooster->param_.v = v;
  pbooster->param_.J = J;
  pbooster->param_.ns = node_size;
  
  pbooster->param_.ratio_si_ = ratio_si;
  pbooster->param_.ratio_fi_ = ratio_fi;
  pbooster->param_.weight_ratio_si_ = weight_ratio_si;

  pbooster->train(&tr);


  /*Output*/
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
  double* pp = mxGetPr(plhs[0]);
  *pp = (long long) pbooster;
}

// Y = predict(dummy, h, X, T);
void predict(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;
  // data X
  MLData te;
  set_X(prhs[2], te.X);
  // T
  int T = (int)mxGetScalar(prhs[3]);

  /* Output */
  int nsample = te.X.rows;
  int K = pbooster->get_class_count();
  plhs[0] = mxCreateNumericMatrix(K, nsample, mxSINGLE_CLASS,mxREAL);
  set_Ymc(plhs[0], te.Y);

  /* Predict */
  pbooster->predict(&te, T);
}

// [NumIter,TrLoss,F,P,GradCls,LossCls] = get(dummy, h);
void get(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;

  /*Output*/
  int T = pbooster->get_num_iter();
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
  double* ptr = mxGetPr(plhs[0]);
  *ptr = double( T );

  plhs[1] = VecDbl_to_mxArray(pbooster->abs_grad_);
  plhs[2] = cvMatDbl_to_mxArray(pbooster->F_);
  plhs[3] = cvMatDbl_to_mxArray(pbooster->p_);
  plhs[4] = cvMatDbl_to_mxArray(pbooster->abs_grad_class_);
  plhs[5] = cvMatDbl_to_mxArray(pbooster->loss_class_);

}

// [node_cc] = get_cc(dummy, h, itree);
void get_cc(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;
  // itree
  int itree = (int)mxGetScalar(prhs[2]);

  // retrieve sample index
  VecInt node_cc;
  pbooster->get_cc(itree, node_cc);

  /*Output*/
  plhs[0] = VecInt_to_mxArray(node_cc);
}

// [node_sc] = get_sc(dummy, h, itree);
void get_sc(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;
  // itree
  int itree = (int)mxGetScalar(prhs[2]);

  // retrieve sample index
  VecInt node_sc;
  pbooster->get_sc(itree, node_sc);

  /*Output*/
  plhs[0] = VecInt_to_mxArray(node_sc);
}

// [is_leaf] = get_is_leaf(dummy, h, itree);
void get_is_leaf(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;
  // itree
  int itree = (int)mxGetScalar(prhs[2]);

  // retrieve sample index
  VecInt is_leaf;
  pbooster->get_is_leaf(itree, is_leaf);

  /*Output*/
  plhs[0] = VecInt_to_mxArray(is_leaf);
}

// delete(dummy, h);
void del(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input */
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pt = (booster_t*) p;

  /* Delete */
  pt->~pAOSOLogitBoostV2();

  /* Output */
  return;
}

// [Nodes, splits, leaves] = save(dummy, h, i);
void save(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input. IMPORTANT: start from index 1 */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  booster_t* pbooster = (booster_t*) p;  


//  //get i
//  int i = (int)mxGetScalar(prhs[2]); 
//  i--;
//
//  /*Output*/
//  plhs[0] = cvMatInt_to_mxArray(pbooster->stor_Trees_[i].nodes_);
//  plhs[1] = cvMatDbl_to_mxArray(pbooster->stor_Trees_[i].splits_);
//  plhs[2] = cvMatDbl_to_mxArray(pbooster->stor_Trees_[i].leaves_);

  //  TODO
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);  
  plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL); 
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL); 
}

// entry point
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  char* str;
  str = mxArrayToString(prhs[0]);

  if ( 0 == strcmp(str,"train") ) {
    train(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"get") ) {
    get(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"get_cc") ) {
    get_cc(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"get_sc") ) {
    get_sc(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"get_is_leaf") ) {
    get_is_leaf(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"predict") ) {
    predict(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"delete") ) {
    del(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"save") ) {
    save(nlhs,plhs, nrhs,prhs);
  }
  else {
    mexErrMsgTxt("pSampVTLogitBoost_mex::unknown option.");
  }
}

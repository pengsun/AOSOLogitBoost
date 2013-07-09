#include "mex.h"
#include "MLData.hpp"
#include "AOSOLogitBoost.hpp"
#include "utilCPP.h"

using namespace cv;

// h = train(dummy, X,Y, var_cat_mask, T,J,v, node_size);
void train(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input. IMPORTANT: start from index 1 */
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


  /* train */
  tr.problem_type = PROBLEM_CLS;
  tr.preprocess();

  AOSOLogitBoost* pbooster = new AOSOLogitBoost;
  pbooster->param_.T = T;
  pbooster->param_.v = v;
  pbooster->param_.J = J;
  pbooster->param_.ns = node_size;
  pbooster->train(&tr);
  
  pbooster->convertToStorTrees();

  /*Output*/
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
  double* pp = mxGetPr(plhs[0]);
  *pp = (long long) pbooster;
}




// [NumIter,TrLoss,F,P] = get(dummy, h);
void get(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input. IMPORTANT: start from index 1 */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  AOSOLogitBoost* pbooster = (AOSOLogitBoost*) p;

  /*Output*/
  plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
  double* ptr = mxGetPr(plhs[0]);
  *ptr = double( pbooster->get_num_iter() );

   plhs[1] = VecDbl_to_mxArray(pbooster->abs_grad_);
   plhs[2] = cvMatDbl_to_mxArray(pbooster->F_);
   plhs[3] = cvMatDbl_to_mxArray(pbooster->p_);
}

// Y = predict(dummy, h, X, T);
void predict(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input. IMPORTANT: start from index 1 */
  // get pointer
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  AOSOLogitBoost* pbooster = (AOSOLogitBoost*) p;
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
// delete(dummy, h);
void del(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[]) 
{
  /* Input. IMPORTANT: start from 1 index */
  double *ptmp = (double*)mxGetData(prhs[1]);
  long long p = (long long) ptmp[0];
  AOSOLogitBoost* pt = (AOSOLogitBoost*) p;

  /* Delete */
  pt->~AOSOLogitBoost();

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
  AOSOLogitBoost* pbooster = (AOSOLogitBoost*) p;  
  

  //get i
  int i = (int)mxGetScalar(prhs[2]); 
  i--;

  /*Output*/
   plhs[0] = cvMatInt_to_mxArray(pbooster->stor_Trees_[i].nodes_);
   plhs[1] = cvMatDbl_to_mxArray(pbooster->stor_Trees_[i].splits_);
   plhs[2] = cvMatDbl_to_mxArray(pbooster->stor_Trees_[i].leaves_);
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
  else if ( 0 == strcmp(str,"predict") ) {
    predict(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str,"delete") ) {
    del(nlhs,plhs, nrhs,prhs);
  }
  else if ( 0 == strcmp(str, "save") ) {
    save(nlhs,plhs, nrhs,prhs);
  }
  else {
    mexErrMsgTxt("LogitBoost_mex::unknown option.");
  }
}
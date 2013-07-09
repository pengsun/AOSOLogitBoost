#include "MLData.hpp"
#include <limits>

using namespace std;
using namespace cv;

MLData::MLData()
{
  problem_type = PROBLEM_REG;
  cls_count_ = -1;
}

// helper class for sorting sample indices
template<typename Idx_t> // Idx_t: unsigned short or int
class LessComp
{
public:
  LessComp (Mat_<float>& _X, int _ivar) {
    X_ = &_X;
    ivar_ = _ivar;
  }

  bool operator() (const Idx_t& idx1, const Idx_t& idx2) {
    return ( X_->at<float>(int(idx1), ivar_) < 
             X_->at<float>(int(idx2), ivar_) );
  }
protected:
  int ivar_;
  Mat_<float> *X_;
};

void MLData::preprocess()
{
  // list for presorting: 16bit or 32bit?
  int nsample = X.rows;
  int nvar = X.cols;
  if (nsample < USHRT_MAX) {
    is_idx16 = true;
    var_num_sidx16.resize(nvar);
  }
  else {
    is_idx16 = false;
    var_num_sidx32.resize(nvar);
  }
  var_cat_count.resize(nvar);
  for (int i = 0; i < var_cat_count.size(); ++i) {
    var_cat_count[i] = -1;
  }
  
  // treat each variable. VAR_CAT: max count; VAR_NUM: sorting
  for (int ivar = 0; ivar < nvar; ++ivar ) {
    if (var_type[ivar]==VAR_CAT) {
      // TODO: assure non empty category!!!
      double maxval,dummy;
      minMaxIdx(X.col(ivar),&dummy,&maxval);
      CV_Assert(maxval >= 0);
      var_cat_count[ivar] = int(maxval+1); // ** 0-base **
    }
    else { // var_type[i]==VAR_NUM: pre sort samples
      if (is_idx16) {
        // initialize
        VecIdx16& vs = var_num_sidx16[ivar];
        vs.resize(nsample);
        for (int i = 0; i < nsample; ++i) {
          vs[i] = (unsigned short)i;
        }
        // sort the indices
        std::sort(vs.begin(),vs.end(), 
                  LessComp<unsigned short>(X,ivar));
      }
      else {
        // initialize
        VecIdx32& vs = var_num_sidx32[ivar];
        vs.resize(nsample);
        for (int i = 0; i < nsample; ++i) {
          vs[i] = (int)i;
        }
        // sort the indices
        std::sort(vs.begin(),vs.end(), 
                  LessComp<int>(X,ivar));
      }
    }
  } // for ivar

  // get class count
  calc_class_count();
}

// class count. -1 if regression problem
int MLData::get_class_count()
{
  if (problem_type==PROBLEM_REG)
    return -1;
  else  // problem_type==PROBLEM_CLS
    return cls_count_;  
}

void MLData::calc_class_count()
{
  if (problem_type==PROBLEM_REG)
    return;

  int N = Y.rows;
  for (int i = 0; i < N; ++i) {
    int cls = int(Y.at<float>(i));
    if (cls_count_ < cls) 
      cls_count_ = cls;
  }
  cls_count_ += 1; // 0-base!!!
}

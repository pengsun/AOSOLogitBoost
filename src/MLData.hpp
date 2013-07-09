#ifndef MLData_H_
#define MLData_H_

#include <opencv2/core/core.hpp>

enum VAR_TYPE
{
  VAR_CAT = 1,
  VAR_NUM = 0
};

enum PROBLEM_TYPE
{
  PROBLEM_CLS = 0, // classification
  PROBLEM_REG = 1   // regression
};

typedef std::vector<unsigned short> VecIdx16;
typedef std::vector<VecIdx16> VecVecIdx16;
typedef std::vector<int> VecIdx32;
typedef std::vector<VecIdx32> VecVecIdx32;

class MLData
{
public:
  const static int MAX_VAR_CAT = 64; // maximum #category
public:
  MLData ();
  void preprocess ();
  int get_class_count ();

public:
  cv::Mat_<float> X; // examples
  cv::Mat_<float> Y; // response
  PROBLEM_TYPE problem_type;
  std::vector<VAR_TYPE> var_type;
  
  // #category for categorical variable.
  // -1 if numerical variable
  std::vector<int> var_cat_count; 
  bool is_idx16;            // true if #sample < 65536
  VecVecIdx16 var_num_sidx16; // sorted indices for numerical variable
  VecVecIdx32 var_num_sidx32;

protected:
  void calc_class_count();
  int cls_count_;

};



#endif
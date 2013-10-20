#include "pAOSOLogitBoostV2.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

float X1[] = {
  0.1f,  0.0f,
  0.41f, 0.0f,
  0.73f, 0.0f,
  0.61f, 0.0f,
  0.93f, 0.0f,
  0.82f, 1.0f
};
float Y1[] = {
  1.0f,
  1.0f,
  1.0f,
  2.0f,
  2.0f,
  3.0f
};


float X2[] = {
  0.1f, 0.2f,
  0.2f, 0.3f,
  0.6f, 0.3f,
  0.7f, 0.2f,
  0.1f, 0.4f,
  0.2f, 0.6f
};
float Y2[] = {
  0.0f,
  0.0f,
  1.0f,
  1.0f,
  2.0f,
  2.0f  
};

float X3[] = {
  0.1f,  0.0f,
  0.41f, 0.0f,
  0.82f, 1.0f,
  0.73f, 0.0f,
  0.61f, 0.0f,
  0.93f, 0.0f
};
float Y3[] = {
  .0f,
  .0f,
  2.0f,
  .0f,
  1.0f,
  1.0f  
};

int main ()
{
  MLData tr;
  //tr.X = Mat(6,2,CV_32FC1,X1);
  //tr.Y = Mat(6,1,CV_32FC1,Y1);
  tr.X = Mat(6,2,CV_32FC1,X2);
  tr.Y = Mat(6,1,CV_32FC1,Y2);  
  tr.var_type.resize(2);
  tr.var_type[0] = VAR_NUM;
  tr.var_type[1] = VAR_NUM;
  tr.problem_type = PROBLEM_CLS;
  tr.preprocess();

  // AOTO Boost
  AOSOLogitBoostV2 ab;
  ab.param_.J = 4;
  ab.param_.T = 3000;
  ab.param_.v = 0.1;
  ab.param_.ns = 1;
  ab.train(&tr);

  MLData te;
  te.X = tr.X;
  ab.predict(&te, 2);
  cout << "predictY = " << te.Y << endl;

  MLData te2;
  te2.X = tr.X;
  ab.predict(&te2, 6);
  cout << "predictY = " << te2.Y << endl;

  int TT = -1;
  TT = ab.get_num_iter();
  cout << endl;
  cout << "TT = " << TT << endl << endl;
  cout << "train loss = " << ab.get_train_loss() << endl;

  MLData te3;
  te3.X = tr.X;
  ab.predict(&te3);
  cout << "predictY = " << te3.Y << endl;

  return 0;
}
#ifndef AOTOBoostSol2Sel2gain_h__
#define AOTOBoostSol2Sel2gain_h__

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>

// data shared by tree and booster
struct AOSODatagain {
  MLData* data_cls_;
  cv::Mat_<double> *F_, *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
};

// Vector for index
typedef std::vector<int> VecIdx;

// Split descriptor
struct AOSOSplitgain {
public:
  AOSOSplitgain ();
  void reset ();

  int var_idx_; // variable for split
  VAR_TYPE var_type_; // variable type

  float threshold_; // for numeric variable 
  std::bitset<MLData::MAX_VAR_CAT> subset_; // mask for category variable, maximum 64 #category 

  double this_gain_;
  double expected_gain_;
  double left_node_gain_, right_node_gain_;
};

// AOTO Node. Vector value
struct AOSONodegain {
public:
  AOSONodegain ();
  AOSONodegain (int _id);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  double fit_val_; // fitness value, i.e., the score f
  int cls1_, cls2_; // first/second class

  int id_; // node ID. 0 for root node
  AOSONodegain *parent_, *left_, *right_; //
  AOSOSplitgain split_;

  VecIdx sample_idx_; // for training
};

// Node Comparator: the less the expected gain, the less the node
struct AOSONodegainLess {
  bool operator () (const AOSONodegain* n1, const AOSONodegain* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for aoto node
typedef std::priority_queue<AOSONodegain*, 
                            std::vector<AOSONodegain*>, 
                            AOSONodegainLess>
                            QueAOSONodegain;
// Adaptive One-to-One Solver
struct AOSOSolver {
  static const double MAXGAMMA;

  static void find_best_two (AOSODatagain* _data, VecIdx& vi,
                             int& cls1, int& cls2);

  AOSOSolver (AOSODatagain* _data, int _cls1, int _cls2);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);
  
  void calc_gamma (double& gamma); 
  void calc_gain (double& gain);
  

protected:
  double mg_[2], h_[2];
  double hh_;
  AOSODatagain* data_;
  int cls1_,cls2_;
};

// AOTO Tree
class AOSOTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    Param ();
  };
  Param param_;

public:
  void split( AOSODatagain* _data );
  void fit ( AOSODatagain* _data );

  AOSONodegain* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

protected:
  void clear ();
  void creat_root_node (AOSODatagain* _data);

  virtual bool find_best_candidate_split (AOSONodegain* _node, AOSODatagain* _data);
  virtual bool find_best_split_num_var (AOSONodegain* _node, AOSODatagain* _data, int _ivar, 
                                        int _cls1, int _cls2);
  void make_node_sorted_idx(AOSONodegain* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( AOSONodegain* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right);

  bool can_split_node (AOSONodegain* _node);
  bool split_node (AOSONodegain* _node, AOSODatagain* _data);
  void calc_gain (AOSONodegain* _node, AOSODatagain* _data);
  virtual void fit_node (AOSONodegain* _node, AOSODatagain* _data);

public:
  std::list<AOSONodegain> nodes_; // all nodes
protected:
  QueAOSONodegain candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  AOSOSplitgain cb_split_, cvb_split_;
  int K_;
};

// the struct to save Tree into files:static storage tree
struct StaticStorTree{
	cv::Mat_<int> nodes_;
	cv::Mat_<double> splits_;
	cv::Mat_<double> leaves_;
};


// AOTO Boost
class AOSOLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAXF;

public:
  struct Param{
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
  };
  Param param_;
  
public:
  void train (MLData* _data);

  void predict (MLData* _data);
  virtual void predict (float* _sapmle, float* _score);
  void predict (MLData* _data, int _Tpre);
  virtual void predict (float* _sapmle, float* _score, int _Tpre);

  int get_class_count ();
  int get_num_iter ();
  double get_train_loss ();
  void calc_grad( int t );
  void convertToStorTrees();


protected:
  void train_init (MLData* _data);
  void update_F(int t);
  void calc_p ();
  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  bool should_stop (int t);
  void convert(AOSONodegain * _root_Node, 
	  StaticStorTree& _sta_Tree, int& _leafId, int& _splitId);

protected:
  int K_; // class count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  AOSODatagain aotodata_;
  std::vector<AOSOTree> trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class

public:
    cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
	std::vector<double> abs_grad_; // total gradient.
	std::vector<StaticStorTree> stor_Trees_;

};
#endif // AOTOBoostSol2Sel2_h__
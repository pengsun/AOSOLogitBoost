#include "pAOSOLogitBoost.hpp"

using namespace std;
using namespace cv;

// Implementation of AOSOSplitgain
AOSOSplitgain::AOSOSplitgain()
{
  reset();
}

void AOSOSplitgain::reset()
{
  var_idx_ = -1;
  threshold_ = FLT_MAX;
  subset_.reset();

  this_gain_ = -1;
  expected_gain_ = -1;
  left_node_gain_ = right_node_gain_ = -1;
}

// Implementation of AOSONodegain
AOSONodegain::AOSONodegain()
{
  id_ = 0;
  parent_ = left_ = right_ = 0;
  fit_val_ = 0;
  cls1_ = cls2_ = -1;
}

AOSONodegain::AOSONodegain( int _id )
{
  id_ = _id;
  parent_ = left_ = right_ = 0;
  fit_val_ = 0;
  cls1_ = cls2_ = -1;
}

int AOSONodegain::calc_dir( float* _psample )
{
  float _val = *(_psample + split_.var_idx_);

  int dir = 0;
  if (split_.var_type_==VAR_CAT) {
    // TODO: raise an error
    /*
    int tmp = int(_val);
    dir = ( split_.subset_[tmp] == true ) ? (-1) : (+1);
    */
  }
  else { // split_.var_type_==VAR_NUM
    dir = (_val < split_.threshold_)? (-1) : (+1); 
  }

  return dir;
}
// Implementation of AOSOSolver
const double AOSOSolver::MAXGAMMA = 5;
void AOSOSolver::find_best_two(
  AOSODatagain* _dt, VecIdx& vidx, int& c1, int& c2 )
{
  // initialize
  c1 = 0; c2 = 1;

  // update mg for all samples
  vector<double> mg;
  int K = _dt->data_cls_->get_class_count();
  mg.assign(K,0);
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    int yi = int( _dt->data_cls_->Y.at<float>(idx) );
    double* ptr_pi = _dt->p_->ptr<double>(idx); 

    for (int k = 0; k < K; ++k) {
      double pik = *(ptr_pi+k);

      mg[k] += ( (yi==k)? (1-pik):(-pik) );
    } // for k
  } // for it

  // find the first
  typedef vector<double>::iterator iter_t;
  iter_t it_beg = mg.begin(), it_end = mg.end();
  iter_t it_max = it_beg;
  for (iter_t it = mg.begin(); it != it_end; ++it) {
    if (*it_max < *it)
      it_max = it;
  }
  c1 = (it_max - it_beg);

  // update h
  vector<double> h, hh;
  h.assign(K,0); hh.assign(K,0);
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    int yi = int( _dt->data_cls_->Y.at<float>(idx) );
    double* ptr_pi = _dt->p_->ptr<double>(idx); 

    double pib = *(ptr_pi + c1); // b for "base class" (a.k.a cls1) 
    for (int k = 0; k < K; ++k) {
      double pik = *(ptr_pi+k);
      h[k]  += ( pik*(1-pik) );

      if (k==c1) continue;
      hh[k] += ( pib*pik );
    } // for k
  } // for it

  // find the second
  double gain_max = -1;
  double mg1 = mg[c1];
  double h11 = h[c1];
  for (int k = 0; k < K; ++k) {
    if (k == c1) continue;  

    double mg2 = mg[k]; 
    double h22 = h[k], h12 = hh[k];

    double mg = mg1 - mg2;
    double h  = h11 + h22 + 2*h12;
    if (h==0) h = 1;
    double gain = mg*mg/h;

    if (gain > gain_max) {
      c2 = k;
      gain_max = gain;
    } // if
  } // for k

}

AOSOSolver::AOSOSolver( AOSODatagain*  _data, int _cls1, int _cls2)
{ 
  data_ = _data;
  
  mg_[0] = mg_[1] = 0;
  h_[0] = h_[1] = 0;
  hh_ = 0;

  cls1_ = _cls1;
  cls2_ = _cls2;
}

void AOSOSolver::update_internal( VecIdx& vidx )
{
  int K = data_->data_cls_->get_class_count();
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    int yi = int( data_->data_cls_->Y.at<float>(idx) );
    double* ptr_pi = data_->p_->ptr<double>(idx); 
    double p1 = *(ptr_pi+cls1_), p2 = *(ptr_pi+cls2_);

    mg_[0] += ( (yi==cls1_)? (1-p1):(-p1) );
    h_[0]  += ( p1*(1-p1) );

    mg_[1] += ( (yi==cls2_)? (1-p2):(-p2) );
    h_[1]  += ( p2*(1-p2) );

    hh_    += p1*p2;
  } // for it
}

void AOSOSolver::update_internal_incre( int idx )
{
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx); 
  double p1 = *(ptr_pi+cls1_), p2 = *(ptr_pi+cls2_);

  mg_[0] += ( (yi==cls1_)? (1-p1):(-p1) );
  h_[0]  += ( p1*(1-p1) );

  mg_[1] += ( (yi==cls2_)? (1-p2):(-p2) );
  h_[1]  += ( p2*(1-p2) );

  hh_    += p1*p2;
}

void AOSOSolver::update_internal_decre( int idx )
{
  int K = data_->data_cls_->get_class_count();
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx); 
  double p1 = *(ptr_pi+cls1_), p2 = *(ptr_pi+cls2_);

  mg_[0] -= ( (yi==cls1_)? (1-p1):(-p1) );
  h_[0]  -= ( p1*(1-p1) );

  mg_[1] -= ( (yi==cls2_)? (1-p2):(-p2) );
  h_[1]  -= ( p2*(1-p2) );

  hh_    -= p1*p2;
}

void AOSOSolver::calc_gamma( double& gamma )
{
  double mg1 = mg_[0], mg2 = mg_[1];
  double h1 = h_[0], h2 = h_[1];

  double mg = mg1-mg2;
  double h = h1 + h2 + 2*hh_;
  if (h==0) h = 1;

  // clap to the range [-MAX,MAX]
  gamma = mg/h;
  if (gamma<-MAXGAMMA) gamma = -MAXGAMMA;
  else if (gamma>MAXGAMMA) gamma = MAXGAMMA; 
}

void AOSOSolver::calc_gain( double& gain )
{
  double mg1 = mg_[0], mg2 = mg_[1];
  double h1 = h_[0], h2 = h_[1];

  double mg = mg1 - mg2;
  double h  = h1 + h2 + 2*hh_;
  if (h==0) h = 1;

  gain = mg*mg/(2*h);
}



// Implementation of best_split_finder (helper class)
best_split_finder::best_split_finder(AOSOTree *_tree, AOSONodegain *_node, AOSODatagain *_data, int cls1, int cls2)
{
  this->tree_ = _tree;
  this->node_ = _node;
  this->data_ = _data;
  this->cls1_ = cls1;
  this->cls2_ = cls2;

  this->cb_split_.reset();
}

best_split_finder::best_split_finder (const best_split_finder &f, cv::Split)
{
  this->tree_ = f.tree_;
  this->node_ = f.node_;
  this->data_ = f.data_;
  this->cls1_ = f.cls1_;
  this->cls2_ = f.cls2_;

  this->cb_split_ = f.cb_split_;
}

void best_split_finder::operator() (const cv::BlockedRange &r)
{

  // for each variable, find the best split
  for (int vi = r.begin(); vi != r.end(); ++vi) {
    AOSOSplitgain the_split;
    the_split.reset();
    bool ret;
    ret = tree_->find_best_split_num_var(node_, data_, vi, 
      cls1_, cls2_, 
      the_split);
    
    // update the cb_split (currently best split)
    if (!ret) continue; // nothing found
    if (the_split.expected_gain_ > cb_split_.expected_gain_) {
      cb_split_ = the_split;
    } // if
  } // for vi
}

void best_split_finder::join (best_split_finder &rhs)
{
  if ( rhs.cb_split_.expected_gain_ > (this->cb_split_.expected_gain_) ) {
    (this->cb_split_) = (rhs.cb_split_);
  }
}

// Implementation of AOSOTree::Param
AOSOTree::Param::Param()
{
  max_leaves_ = 2;
  node_size_ = 5;
}
// Implementation of AOSOTree
void AOSOTree::split( AOSODatagain* _data )
{
  // clear
  clear();
  K_ = _data->data_cls_->get_class_count();

  // root node
  creat_root_node(_data);
  candidate_nodes_.push(&nodes_.front());
  AOSONodegain* root = candidate_nodes_.top(); 
  find_best_candidate_split(root, _data);
  int nleaves = 1;

  // split recursively
  while ( nleaves < param_.max_leaves_ &&
          !candidate_nodes_.empty() )
  {
    AOSONodegain* cur_node = candidate_nodes_.top(); // the most prior node
    candidate_nodes_.pop();
    --nleaves;

    if (!can_split_node(cur_node)) { // can not split, make it a leaf
      ++nleaves;
      continue;
    }


    split_node(cur_node,_data);
    VecIdx tmp;
    tmp.swap(cur_node->sample_idx_); // release memory.
    // no longer used in later splitting

    // find best split for the two newly created nodes
    find_best_candidate_split(cur_node->left_, _data);
    candidate_nodes_.push(cur_node->left_);
    ++nleaves;

    find_best_candidate_split(cur_node->right_, _data);
    candidate_nodes_.push(cur_node->right_);
    ++nleaves;
  }
}

void AOSOTree::fit( AOSODatagain* _data )
{

  // fitting node data for each leaf
  std::list<AOSONodegain>::iterator it;
  for (it = nodes_.begin(); it != nodes_.end(); ++it) {
    AOSONodegain* nd = &(*it);

    if (nd->left_!=0) { // not a leaf
      continue;
    } 

    fit_node(nd,_data);

    // release memory.
    // no longer used in later splitting
    VecIdx tmp;
    tmp.swap(nd->sample_idx_);
  }
}

AOSONodegain* AOSOTree::get_node( float* _sample)
{
  AOSONodegain* cur_node = &(nodes_.front());
  while (true) {
    if (cur_node->left_==0) break; // leaf reached 

    int dir = cur_node->calc_dir(_sample);
    AOSONodegain* next = (dir==-1) ? (cur_node->left_) : (cur_node->right_);
    cur_node = next;
  }
  return cur_node;
}
void AOSOTree::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void AOSOTree::predict( float* _sample, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update the two class
  AOSONodegain* nd;
  nd = get_node(_sample);
  int cls1 = nd->cls1_, cls2 = nd->cls2_;
  double gamma = nd->fit_val_;

  *(_score + cls1) = static_cast<float>( gamma );
  *(_score + cls2) = static_cast<float>( -gamma );
}

void AOSOTree::clear()
{
  nodes_.clear();
  while(!candidate_nodes_.empty())
	  candidate_nodes_.pop();
}

void AOSOTree::creat_root_node( AOSODatagain* _data )
{
  nodes_.push_back(AOSONodegain(0));
  AOSONodegain* root = &(nodes_.back());

  // samples in node
  int N = _data->data_cls_->X.rows;
  root->sample_idx_.resize(N);
  for (int i = 0; i < N; ++i) {
    root->sample_idx_[i] = i;
  }

  // loss
  this->calc_gain(root, _data);
}

bool AOSOTree::find_best_candidate_split( AOSONodegain* _node, AOSODatagain* _data )
{
  bool found_flag = false;
  MLData* data_cls = _data->data_cls_;

  // the best class-pair
  int cls1, cls2;
  AOSOSolver::find_best_two(_data,_node->sample_idx_, cls1,cls2);

  // the range (beggining/ending variable)
  int nvar = data_cls->X.cols;
  cv::BlockedRange br(0,nvar,1);

  // do the search in parallel
  best_split_finder bsf(this,_node,_data, cls1,cls2);
  cv::parallel_reduce(br, bsf);

  // update node's split
  _node->split_ = bsf.cb_split_;
  return true; // TODO: Check if this is reasonable

  //cvb_split_.reset(); cb_split_.reset();
  //for (int ivar = 0; ivar < nvar; ++ivar) {
  //  bool ret;
  //  if (data_cls->var_type[ivar] == VAR_CAT) {
  //    // TODO: raise an error
  //  }
  //  else { // VAR_NUM
  //    ret = find_best_split_num_var(_node, _data,ivar, cls1, cls2);
  //  }

  //  if (!ret) continue; // nothing found

  //  if (cvb_split_.expected_gain_ > cb_split_.expected_gain_) {
  //    found_flag = true;
  //    cb_split_ = cvb_split_;   
  //  }

  //}

  //if (found_flag) {
  //  _node->split_ = cb_split_;
  //  return true;
  //}
  //else {
  //  _node->split_.var_idx_ = -1;
  //  return false;
  //}
}

bool AOSOTree::find_best_split_num_var( 
  AOSONodegain* _node, AOSODatagain* _data, int _ivar, 
  int _cls1, int _cls2, AOSOSplitgain &cb_split)
{
  VecIdx node_sample_si;
  MLData* data_cls = _data->data_cls_;
  make_node_sorted_idx(_node,data_cls,_ivar,node_sample_si);
  int ns = node_sample_si.size();
  if (ns < 1) return false;

  // initialize
  AOSOSolver sol_left(_data,_cls1,_cls2), sol_right(_data,_cls1,_cls2);
  sol_right.update_internal(node_sample_si);

  // scan each possible split 
  double best_gain = -1, best_gain_left = -1, best_gain_right = -1;
  int best_i = -1;
  for (int i = 0; i < ns-1; ++i) {  // ** excluding null and all **
    int idx = node_sample_si[i];
    sol_left.update_internal_incre(idx);
    sol_right.update_internal_decre(idx);

    // skip if overlap
    int idx1 = idx;
    float x1 = data_cls->X.at<float>(idx1, _ivar);
    int idx2 = node_sample_si[i+1];
    float x2 = data_cls->X.at<float>(idx2, _ivar);
    if (x1==x2) continue; // overlap

    // check left & right
    double gL;
    sol_left.calc_gain(gL);
    double gR;
    sol_right.calc_gain(gR);
    
    double g = gL + gR;
    if (g > best_gain) {
      best_i = i;
      best_gain = g;
      best_gain_left = gL; best_gain_right = gR;
    } // if
  } // for i

  // set output
  return set_best_split_num_var(
    _node, data_cls, _ivar,
    node_sample_si,
    best_i, best_gain, best_gain_left, best_gain_right,
    cb_split);
}

void AOSOTree::make_node_sorted_idx( AOSONodegain* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node )
{
  VecIdx16 sam_idx16;
  VecIdx32 sam_idx32;
  if (_data->is_idx16) 
    sam_idx16 = _data->var_num_sidx16[_ivar];
  else
    sam_idx32 = _data->var_num_sidx32[_ivar];

  // mask for samples in _node
  int N = _data->X.rows;
  vector<bool> mask(N,false); 
  int nn = _node->sample_idx_.size();
  for (int i = 0; i < nn; ++i) {
    int idx = _node->sample_idx_[i];
    mask[idx] = true;
  }

  // copy the sorted indices for samples in _node
  sorted_idx_node.reserve(nn);
  if (_data->is_idx16) {
    for (int i = 0; i < N; ++i) {
      int ix = int(sam_idx16[i]);
      if (mask[ix]) 
        sorted_idx_node.push_back( ix );
    }
  }
  else {
    for (int i = 0; i < N; ++i) {
      int ix = int(sam_idx32[i]);
      if (mask[ix])
        sorted_idx_node.push_back(ix);
    }
  }  
}

bool AOSOTree::set_best_split_num_var( 
  AOSONodegain* _node, MLData* _data, int _ivar, 
  VecIdx& node_sample_si, 
  int best_i, double best_gain, double best_gain_left, double best_gain_right,
  AOSOSplitgain &cb_split)
{
  if (best_i==-1) return false; // fail to find...

  // set gains
  double this_gain = _node->split_.this_gain_;
  cb_split.expected_gain_ = best_gain - this_gain; 
  cb_split.left_node_gain_ = best_gain_left;
  cb_split.right_node_gain_ = best_gain_right;

  // set split
  cb_split.var_idx_ = _ivar;
  cb_split.var_type_ = _data->var_type[_ivar]; 
  int idx1 = node_sample_si[best_i];
  float x1 = _data->X.at<float>(idx1, _ivar);
  int idx2 = node_sample_si[best_i+1];
  float x2 = _data->X.at<float>(idx2, _ivar);
  if (x2>x1)
    cb_split.threshold_ = (x1+x2)/2;
  else
    return false; // all samples overlap, fail to split...

  return true;  
}

bool AOSOTree::can_split_node( AOSONodegain* _node )
{
  bool flag = true;
  int nn = _node->sample_idx_.size();
  int idx = _node->split_.var_idx_;
  return (nn > param_.node_size_    && // large enough node size
          idx != -1);                  // has candidate split  
}

bool AOSOTree::split_node( AOSONodegain* _node, AOSODatagain* _data )
{
  // create left and right node
  AOSONodegain tmp1(nodes_.size());
  nodes_.push_back(tmp1);
  _node->left_ = &(nodes_.back());
  _node->left_->parent_ = _node;

  AOSONodegain tmp2(nodes_.size());
  // 
  nodes_.push_back(tmp2);
  _node->right_ = &(nodes_.back());
  _node->right_->parent_ = _node;

  // send each sample to left/right node
  int nn = _node->sample_idx_.size();
  CV_Assert(_node->split_.var_idx_>-1);
  MLData* data_cls = _data->data_cls_;
  for (int i = 0; i < nn; ++i) {
    int idx = _node->sample_idx_[i];

    float* p = (float*)data_cls->X.ptr(idx);
    int dir = _node->calc_dir(p);
    if (dir == -1) 
      _node->left_->sample_idx_.push_back(idx);
    else 
      _node->right_->sample_idx_.push_back(idx);
  }

  // initialize the node gain
  this->calc_gain(_node->left_, _data);
  this->calc_gain(_node->right_, _data);

  return true;
}

void AOSOTree::calc_gain(AOSONodegain* _node, AOSODatagain* _data)
{
  int cls1, cls2;
  AOSOSolver::find_best_two(_data,_node->sample_idx_,
    cls1,cls2);
  AOSOSolver sol(_data,cls1,cls2);
  sol.update_internal(_node->sample_idx_);
  double gain;
  sol.calc_gain(gain);
  _node->split_.this_gain_ = gain;
}


void AOSOTree::fit_node( AOSONodegain* _node, AOSODatagain* _data )
{
  int nn = _node->sample_idx_.size();
  if (nn<=0) return;

  int cls1, cls2;
  AOSOSolver::find_best_two(_data, _node->sample_idx_, cls1,cls2);
  AOSOSolver sol(_data, cls1,cls2);
  sol.update_internal(_node->sample_idx_);
  double gamma;
  sol.calc_gamma(gamma);

  _node->cls1_ = cls1;
  _node->cls2_ = cls2;
  _node->fit_val_ = gamma;
}



// Implementation of AOSOLogitBoost
const double AOSOLogitBoost::EPS_LOSS = 1e-16;
const double AOSOLogitBoost::MAXF = 100;
void AOSOLogitBoost::train( MLData* _data )
{
  train_init(_data);

  for (int t = 0; t < param_.T; ++t) {
    trees_[t].split(&aotodata_);
    trees_[t].fit(&aotodata_);

    update_F(t);
    calc_p();
    calc_loss(_data);
    calc_loss_iter(t);
	calc_grad(t);

    NumIter_ = t + 1;
    if ( should_stop(t) ) break;
  } // for t

}

void AOSOLogitBoost::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void AOSOLogitBoost::predict( float* _sapmle, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  } // for k

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = 0; t < NumIter_; ++t) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    } // for k
  } // for t
}

void AOSOLogitBoost::predict( MLData* _data, int _Tpre )
{
  // trees to be used
  if (_Tpre > NumIter_) _Tpre = NumIter_;
  if (_Tpre < 1) _Tpre = 1; // _Tpre in [1,T]
  if (Tpre_beg_ > _Tpre) Tpre_beg_ = 0;

  // initialize predicted score
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);
  
  // initialize internal score if necessary
  if (Tpre_beg_ == 0) {
    Fpre_.create(N,K);
    Fpre_ = 0;
  }

  // for each sample
  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score, _Tpre);
    
    // update score and internal score Fpre_
    double* pp = Fpre_.ptr<double>(i);
    for (int k = 0; k < K; ++k) {
      *(score+k) += *(pp+k);
      *(pp+k) = *(score+k);
    }
  }
  
  // Set the new beginning tree
  Tpre_beg_ = _Tpre;

}

void AOSOLogitBoost::predict( float* _sapmle, float* _score, int _Tpre )
{
  // IMPORTANT: caller should assure the validity of _Tpre

  // initialize 
  for (int k = 0; k < K_; ++k) 
    *(_score+k) = 0;

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = Tpre_beg_; t < _Tpre; ++t ) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    }
  }

}
int AOSOLogitBoost::get_class_count()
{
  return K_;
}

int AOSOLogitBoost::get_num_iter()
{
  return NumIter_;
}

double AOSOLogitBoost::get_train_loss()
{
  if (NumIter_<1) return DBL_MAX;
  return L_iter_.at<double>(NumIter_-1);
}

void AOSOLogitBoost::train_init( MLData* _data )
{
  // class count
  K_ = _data->get_class_count();

  // F, p
  int N = _data->X.rows;
  F_.create(N,K_); 
  F_ = 0;
  p_.create(N,K_); 
  calc_p();

  // Loss
  L_.create(N,1);
  calc_loss(_data);
  L_iter_.create(param_.T,1);

  // iteration for training
  NumIter_ = 0;


  // AOTOData
  aotodata_.data_cls_ = _data;
  aotodata_.F_ = &F_;
  aotodata_.p_ = &p_;
  aotodata_.L_ = &L_;

  // trees
  trees_.clear();
  trees_.resize(param_.T);
  for (int t = 0; t < param_.T; ++t) {
    trees_[t].param_.max_leaves_ = param_.J;
    trees_[t].param_.node_size_ = param_.ns;
  }

  // gradient/delta
  abs_grad_.clear();
  abs_grad_.resize(param_.T);

  // for prediction
  Tpre_beg_ = 0;
}

void AOSOLogitBoost::update_F( int t )
{
  int N = aotodata_.data_cls_->X.rows;
  double v = param_.v;
  vector<float> f(K_);
  for (int i = 0; i < N; ++i) {
    float *psample = aotodata_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);

    double* pF = F_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(pF+k) += (v*f[k]);
      // MAX cap
      if ( *(pF+k) > MAXF ) *(pF+k) = MAXF;
    } // for k
  } // for i
}

void AOSOLogitBoost::calc_p()
{
  int N = F_.rows;
  int K = K_;
  std::vector<double> tmpExpF(K);
  for (int n = 0; n < N; ++n) {
    double tmpSumExpF = 0;
    double* ptrF = F_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      double Fnk = *(ptrF + k);
      double tmp = exp(Fnk);
      tmpExpF[k] = tmp;
      tmpSumExpF += tmp;
    } // for k

    double* ptrp = p_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      // TODO: does it make any sense??
      if (tmpSumExpF==0) tmpSumExpF = 1;
      *(ptrp + k) = double( tmpExpF[k]/tmpSumExpF );
    } // for k
  }// for n  
}

void AOSOLogitBoost::calc_loss( MLData* _data )
{
  const double PMIN = 0.0001;
  int N = _data->X.rows;
  for (int i = 0; i < N; ++i) {
    int yi = int( _data->Y.at<float>(i) );
    double* ptr = p_.ptr<double>(i);
    double pik = *(ptr + yi);

    if (pik<PMIN) pik = PMIN;
    L_.at<double>(i) = (-log(pik));
  }
}

void AOSOLogitBoost::calc_loss_iter( int t )
{
  double sum = 0;
  int N = L_.rows;
  for (int i = 0; i < N; ++i) 
    sum += L_.at<double>(i);

  L_iter_.at<double>(t) = sum;
}

bool AOSOLogitBoost::should_stop( int t )
{
  double loss = L_iter_.at<double>(t);
  return ( (loss<EPS_LOSS) ? true : false );
}

void AOSOLogitBoost::calc_grad( int t )
{
	int N = F_.rows;
	double delta = 0;

	for (int i = 0; i < N; ++i) {
		double* ptr_pi = p_.ptr<double>(i);
		int yi = int( aotodata_.data_cls_->Y.at<float>(i) );

		for (int k = 0; k < K_; ++k) {
			double pik = *(ptr_pi+k);
			if (yi==k) delta += std::abs( 1-pik );
			else       delta += std::abs( -pik );    
		}
	}

	abs_grad_[t] = delta;  
}

void AOSOLogitBoost::convertToStorTrees()
{
	//initailize    
    int i;
    int n = 2*param_.J - 1;
    stor_Trees_.resize(NumIter_);
	for (i=0; i< NumIter_; i++){		
		stor_Trees_[i].nodes_.create(n, 5);// parentID, leftID, rightID, splitID, leafID
		stor_Trees_[i].nodes_ = -1;
		stor_Trees_[i].splits_.create(param_.J-1, 2);
		stor_Trees_[i].splits_ = -1;
		stor_Trees_[i].leaves_.create(param_.J, 3);
		stor_Trees_[i].leaves_ = -1;
	}//for

	//convert
	for (i=0; i<NumIter_; i++)
	{
		int _leafId = 0;
		int _splitId = 0;
		AOSONodegain* root_Node = &( trees_[i].nodes_.front());
		convert(root_Node, stor_Trees_[i], _leafId, _splitId);
	}
}

void AOSOLogitBoost::convert(AOSONodegain * _root_Node, 
	StaticStorTree& _sta_Tree, int& _leafId, int& _splitId)
{
	//convert root node
	int nodeId = _root_Node->id_;
	if (_root_Node->parent_ == NULL)
		_sta_Tree.nodes_.at<int>(nodeId, 0)  = -1;
	else _sta_Tree.nodes_.at<int>(nodeId, 0)  = _root_Node->parent_->id_;

	if (_root_Node->left_ == NULL){ //leaf
		_sta_Tree.nodes_.at<int>(nodeId, 4) = _leafId;
		_sta_Tree.leaves_.at<double>(_leafId, 0) = _root_Node->cls1_;
		_sta_Tree.leaves_.at<double>(_leafId, 1) = _root_Node->cls2_;
		_sta_Tree.leaves_.at<double>(_leafId, 2) = _root_Node->fit_val_;
		_leafId ++;
	}
	else{//internal node
		_sta_Tree.nodes_.at<int>(nodeId, 1)  = _root_Node->left_->id_;
		_sta_Tree.nodes_.at<int>(nodeId, 2)  = _root_Node->right_->id_;
		_sta_Tree.nodes_.at<int>(nodeId, 3) = _splitId;
		_sta_Tree.splits_.at<double>(_splitId, 0) = _root_Node->split_.var_idx_;
		_sta_Tree.splits_.at<double>(_splitId, 1) = _root_Node->split_.threshold_;
		_splitId ++;

		//convert left subtree
		convert(_root_Node->left_, _sta_Tree, _leafId, _splitId);

		//convert right subtree
		convert(_root_Node->right_, _sta_Tree, _leafId, _splitId);
	}
}
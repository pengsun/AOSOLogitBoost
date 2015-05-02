AOSOLogitBoost
==============

Codes for the the so-called AOSO-LogitBoost[3,4], which is an up-to-date (yet state-of-the-art, probably ) variant of LogitBoost[1] but focuses on multi-class classification. For binary classification, it reduces to the original LogitBoost[1] with the robust tree split gain criterion[2]. Onc e you decide that LogitBoost is suitable to your classification problem, just try this AOSO-LogitBoost which typically has lower classification error and faster convergence rate than original LogitBoost. 

## Features
* C++ source codes, with interfaces in Matlab
* Multi threaded implementation (depending on tbb)
* Training speedup by subsampling instances/features (mature best-practice) is supported

## 3rd Party Dependencies
Opencv (opencv_core only), which itself depends on tbb for multi-threading.

## Install
- Compile the mex files. cd to `AOSOLogitBoost/matlab/mex`, run `make_xxx.m` to compile the corresponding `xxx` Boosting algorithm.
  - Remember to modify or add new `settings` m file so that the 3rd party libraries point to the right path. For example, in the first line of `make_pAOSOLogitBoostV2.m` the `settings_yyy.m` is called to set the path for OpenCV and TBB, check the contents and adapt them to your own machine.
  - The compiled mex files will be copied to `AOSOLogitBoost/private` directory. A win32 mex file is already there as the example.
  - If you've compiled OpenCV as dynamic linking, make sure the binaries are on your system path.
- Done! cd to `AOSOLogitBoost/matlab/script_run` and play around!

## Examples
### C++ examples
    TODO
    
### Matlab examples   
The C++ codes are wrapped with Matlab class. Currently we provide the following classes:
* AOSOLogitBoost: Single threaded implementation of AOSO-LogitBoot.
* pAOSOLogitBoost: Multiple threaded implementation of AOSO-LogitBoost.
* pAOSOLogitBoostV2: Multiple threaded implementation of AOSO-LogitBoost, speedup by subsampling instances/features.

See the script files in directory `./matlab/run_script` for various examples. To begin with, here are some simple examples:

#### Example 1. Calling AOSOLogitBoost: 

``` Matlab
%% prepare train/test data. 
% 3-class classification. Features are 2 dimensional. 
% 6 training examples and 3 testing examples. 
Xtr = [... 
  0.1, 0.2; 
  0.2, 0.3; 
  0.6, 0.3; 
  0.7, 0.2; 
  0.1, 0.4; 
  0.2, 0.6... 
 ]; 
Xtr = Xtr'; 
Xtr = single(Xtr); 
% Xtr should be 2X6, single

Ytr = [... 
  0.0; 
  0.0; 
  1.0; 
  1.0; 
  2.0; 
  2.0; 
]; 
Ytr = Ytr'; 
Ytr = single(Ytr); 
% Ytr should be 1X6,single 
% K = 3 classes(0,1,2)
  
Xte = [... 
  0.1, 0.2; 
  0.6, 0.3; 
  0.2, 0.6... 
]; 
Xte = Xte'; 
Xte = single(Xte);

Yte = [... 
  0; 
  1; 
  2; 
]; 
Yte = Yte'; 
Yte = single(Yte);

%% parameters 
T = 2; % #iterations 
v = 0.1; % shrinkage factor 
J = 4; % #terminal nodes 
nodesize = 1; % node size. 1 is suggested 
catmask = uint8([0,0,0,0]); % all features are NOT categorical data 
                            % Currently only numerical data are supported:)
  
%% train 
hboost = AOSOLogitBoost(); % handle 
hboost = train(hboost,... 
  Xtr,Ytr,... 
  'T', T,... 
  'v', v,... 
  'J',J,... 
  'node_size',nodesize,... 
  'var_cat_mask',catmask);
  
%% predict 
F = predict(hboost, Xte); 
% The output F now is a #classes X #test-exmaples matrix. 
% F(k,j) denotes the confidence to predict the k-th class for the j-th test example. 
% Just pick the maximum component of F(:,j) as your prediction for the j-th test example.

%% error and error rate 
[~,yy] = max(F); 
yy = yy - 1; % index should be 0-base 
err_rate = sum(yy~=Yte)/length(Yte) 
```

#### Exmaple 2. Calling pAOSOLogitBoost

Just replace the class `AOSOLogitBoost` in previous example with `pAOSOLogitBoost`, where the leading "p" stands for parallel. See the script files in `Matlab/script_run`.

#### Example 3. Calling pAOSOLogitBoostV2

``` Matlab
%% prepare train/testdata
% ...
% The same with pAOSOLogitBoost, codes omitted here

%% parameters
T = 2; % #iterations
v = 0.1; % shrinkage factor
J = 4; % #terminal nodes
nodesize = 1; % node size. 1 is suggested
catmask = uint8([0,0,0,0]); % all features are NOT categorical data
                            % Currently only numerical data are supported:)
wrs = 0.9; % subsampling instances accouting for 90% weights (denoted by N1)
rs = 0.4;  % subsampling 40% instances (denoted by N2)
           % min(N1,N2) instances will be used at each boosting iteration
rf = 0.6;  % 60% features will be used at each tree node           

%% train    
hboost = pAOSOLogitBoostV2(); % handle
hboost = train(hboost,...
  Xtr,Ytr,...
  'T', T,...
  'v', v,...
  'J',J,...
  'node_size',nodesize,...
  'var_cat_mask',catmask,...
  'wrs',wrs, 'rs',rs,...
  'rf',rf);

%% predict and error rate
% codes omitted here
```

## The method
If you are interested in algorithm's details or concerning how much the improvement is, please refer to [3, 4].

## References
[1] Jerome Friedman, Trevor Hastie and Robert Tibshirani. Additive logistic regression: a statistical view of boosting. Annals of Statistics 28(2), 2000. 337–407.

[2] Ping Li. Robust logitboost and adaptive base class (abc) logitboost, Conference on Uncertainty in Artificial Intelligence (UAI 2010).

[3] Peng Sun, Mark D. Reid, Jie Zhou. AOSO-LogitBoos t: Adaptive One-Vs-One LogitBoost for Multi-Class Problems, International Conference on Machine Learning (ICML 2012).

[4] Peng Sun, Mark D. Reid, Jie Zhou. "An Improved Multiclass LogitBoost Using Adaptive-One-vs-One", Machine Learning (MLJ), 2014, 97(3): 295-326.

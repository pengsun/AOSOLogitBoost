AOSOLogitBoost
==============

Codes for the the so-called AOSO-LogitBoost, which is an up-to-date (yet state-of-the-art, probably ) implementation of Friedman's LogitBoost for multi-class classification. Once you decide that LogitBoost is suitable to your classification problem, just try this AOSO-LogitBoost which typically has lower classification error and faster convergence rate than original LogitBoost. 

Features
--------
* C++ source codes, with interfaces in other languages (say, Matlab).
* Multi threaded implementation (depending on tbb)

Dependencies
------------
Opencv, which itself depending on tbb when using multi threaded features.


Examples
--------
### C++ example
    TODO
    
### Matlab example    
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

### References
Those who are interested in algorithm's details are referred to the paper:
"Peng Sun, Mark D. Reid, Jie Zhou. "AOSO-LogitBoost: Adaptive One-Vs-One LogitBoost for Multi-Class Problems", International Conference on Machine Learning (ICML 2012)"

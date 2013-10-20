%% prepare train/testdata
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
%% predict
F = predict(hboost, Xte);

%% error and error rate
[~,yy] = max(F);
yy = yy - 1; % index should be 0-base
err_rate = sum(yy~=Yte)/length(Yte)
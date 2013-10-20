%%
fn_data = '.\dataset\pendigits.mat';
dir_rst = '.\rst\pAOSOLogitBoostV2\pendigits';
% dir_rst = './';
%%
num_Tpre = 5000;
T = 5000;
cv  = {0.1};
cJ = {20};
cns = {1};
%%% sample
crs = {0.1};
cwrs = {0.95};
%%% feature
crf = {0.2};
%%
h = batch_pAOSOLogitBoostV2();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
% sample
h.cwrs = cwrs;
h.crs = crs;
% feature
h.crf = crf;
run_all_param(h, fn_data, dir_rst);
clear h;

%%
fn_data = '.\dataset\optdigits.mat';
dir_rst = '.\rst\AOSOLogitBoost\optdigits';
% dir_rst = './';
%%
num_Tpre = 2000;
T = 600;
cv = {0.1};
cJ = {20};
cns = {1};

h = batch_AOSOLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
run_all_param(h, fn_data, dir_rst);
clear h;

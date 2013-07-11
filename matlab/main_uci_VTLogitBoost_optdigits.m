%% optdigits
fn_data = 'D:\Users\sp\data\dataset_mat\optdigits.mat';
dir_rst = 'rst_uci\VTLogitBoost\optdigits';

num_Tpre = 3;
T = 3;
cv  = {0.1};
cJ = {8};
cns = {1};

h = uci_VTLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;
%% pendigits
% fn_data = 'D:\Users\sp\data\dataset_mat\pendigits.mat';
% dir_rst = 'rst_uci\VTLogitBoost\pendigits';
% 
% num_Tpre = 2000;
% T = 3;
% cv = {0.1};
% cJ = {20};
% cns = {1};
% 
% h = uci_VTLogitBoost();
% h.num_Tpre = num_Tpre;
% h.T = T;
% h.cv = cv;
% h.cJ = cJ;
% h.cns = cns;
% run_all_param(h, fn_data, dir_rst);
% clear h;


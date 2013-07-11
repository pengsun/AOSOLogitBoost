%%
name_data = 'covertype';
dir_data = 'D:\Users\sp\data\dataset_mat';
fn_data = fullfile(dir_data, [name_data,'.mat']);
dir_rst = fullfile('.\rst\AOSOLogitBoost\',name_data);
% dir_rst = './';
%%
num_Tpre = 2000;
T = 100000;
cv = {0.1};
% cJ = {150,180,210};
% cJ = {300,400,500};
% cJ = {800,1000,2000,3000};
% cJ = {15000};
% cJ = {20000};
cJ = {30000};
cns = {1};

h = batch_AOSOLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;

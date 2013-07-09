%% OpenCV 
% root dir
dir_root = '"D:\Program Files\opencv"';
% dir
cvinc2 = fullfile(dir_root,'\modules\core\include');
linkdird = fullfile(dir_root,'\buildmy\lib');
linkdir = fullfile(dir_root, '\buildmy\lib');
% 
lib2d = 'opencv_core243d';
lib2 = 'opencv_core243';
%% source codes
dir_src = '../../src/';
%% options
tmpld = '-I%s -I%s -L%s -l%s';
opt_cmdd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdird,lib2d);
%% string template
tmpl = '-I%s -I%s -L%s -l%s';
opt_cmd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdir,lib2);
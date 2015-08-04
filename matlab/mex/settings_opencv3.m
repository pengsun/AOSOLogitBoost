%% OpenCV 
% root dir
dir_root = 'D:\CodeWork\libs\opencv300';
% dir
cvinc2 = fullfile(dir_root,'sources\modules\core\include');
linkdird = fullfile(dir_root,'build_msvc2012dll\lib\Debug');
linkdir = fullfile(dir_root, 'build_msvc2012dll\lib\Release');
% 
lib2d = 'opencv_core300d';
lib2 = 'opencv_core300';
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
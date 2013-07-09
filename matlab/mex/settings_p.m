%% OpenCV 
% root dir
dir_root = '"D:\Program Files\opencv"';
% 
cvinc2 = fullfile(dir_root,'\modules\core\include');
linkdird = fullfile(dir_root,'\buildmy\lib');
linkdir = fullfile(dir_root, '\buildmy\lib');
% 
lib2d = 'opencv_core243d';
lib2 = 'opencv_core243';
%% source codes
dir_src = '../../src/';
%% TBB
dir_tbb = '"D:\Program Files\tbb41"';
dir_tbbinc = fullfile(dir_tbb, '\include');
dir_tbblib = fullfile(dir_tbb, '\lib\ia32\vc9');
tbblib = 'tbb';
%% options debug
% don't forget the HAVE_TBB preprocessor for opencv
tmpld = '-I%s -I%s -L%s -l%s -I%s -L%s -l%s -DHAVE_TBB';
opt_cmdd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdird,lib2d,...
  dir_tbbinc,dir_tbblib,tbblib);
%% options release
% don't forget the HAVE_TBB preprocessor for opencv
tmpl = '-I%s -I%s -L%s -l%s -I%s -L%s -l%s -DHAVE_TBB';
opt_cmd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdir,lib2,...
  dir_tbbinc,dir_tbblib,tbblib);
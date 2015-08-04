%% OpenCV 
% root dir
dir_root = 'D:\CodeWork\libs\opencv300';
% 
cvinc2 = fullfile(dir_root,'sources\modules\core\include');
linkdird = fullfile(dir_root,'build_msvc2012dll\lib\Debug');
linkdir = fullfile(dir_root, 'build_msvc2012dll\lib\Release');
% 
lib2d = 'opencv_core300d';
lib2 = 'opencv_core300';
%% source codes
dir_src = '../../src/';
%% TBB
dir_tbb = 'D:\CodeWork\libs\tbb43';
dir_tbbinc = fullfile(dir_tbb, '\include');
dir_tbblib = fullfile(dir_tbb, '\lib\intel64\vc11');
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
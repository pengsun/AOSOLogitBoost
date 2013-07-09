%% root dir
dir_root = 'D:\Users\sp\WorkStudy\OpenCV-2.2.0';
%% dir
cvinc1 = fullfile(dir_root,'modules\ml\include');
cvinc2 = fullfile(dir_root,'\modules\core\include');
linkdird = fullfile(dir_root,'\lib\Debug');
linkdir = fullfile(dir_root, '\lib\Release');
%% 
lib1d = 'opencv_ml220d';
lib2d = 'opencv_core220d';
lib1 = 'opencv_ml220';
lib2 = 'opencv_core220';
%% option
tmpld = '-I%s -I%s -L%s -l%s -L%s -l%s';
opt_cmdd = sprintf(tmpld,...
  cvinc1,cvinc2,...
  linkdird, lib1d, linkdird,lib2d);
%%
tmpl = '-I%s -I%s -L%s -l%s -L%s -l%s';
opt_cmd = sprintf(tmpld,...
  cvinc1,cvinc2,...
  linkdir, lib1, linkdir,lib2);
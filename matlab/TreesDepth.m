% 'M-Noise3', 'M-Noise4', 'M-Noise5', 'M-Noise6', 
% nameSet = {'M-Rotate', ...
%     'pendigits', 'poker100k', 'zipcode', 'isolet'};
%%
nameSet = { 'covertype145k',};
dir_root = '.\rst\AOSOLogitBoost';
J = 20; %num of leaves
%% load data
name = nameSet{1};
dataset = 'T50000_v1.0e-001_J20_ns1.mat';
dataset = fullfile(dir_root,name,dataset);
load(dataset);
%% calculate tree depth
[num, ~] = size(Trees);
[~, n] = size(Trees(1).nodes);
treesDepth = zeros(num, J);
for i=1:num
  j = 0;
  for i1=1:n
    if Trees(i).nodes(5, i1) ~= 0 %find leaves
      j = j+1; %the jth leaves
      p = i1;  %parent node id
      treesDepth(i, j) = 1; %tree depth
      while( Trees(i).nodes(1, p) ~= 0 )
        p = Trees(i).nodes(1, p);
        treesDepth(i, j) =  treesDepth(i, j) + 1;
      end
    end
  end
end

maxDepth = max(treesDepth,[], 2);
avr_maxDepth =  mean(maxDepth);
avr_avrDepth = mean(mean(treesDepth, 2));

%% disp
fprintf('-------------\n');
fprintf('dataset: %s \n', name);
fprintf('average depth: %.4f \n', avr_maxDepth);
fprintf('average avr-depth: %.4f \n\n', avr_avrDepth);

%% save
dataset = 'treeDepth.mat';
dataset = fullfile(dir_root,name,dataset);
save(dataset, 'maxDepth', 'treesDepth', ...
  'avr_maxDepth', 'avr_avrDepth');
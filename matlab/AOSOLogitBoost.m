classdef AOSOLogitBoost
  % Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    ptr;
  end
  
  methods
    function obj = train(obj,X,Y, varargin)
      [var_cat_mask,T,J,v, node_size] = parse_input(varargin{:});
      if (isempty(var_cat_mask))
        nvar = size(X,1);
        var_cat_mask = uint8( zeros(nvar,1) );
      end
      obj.ptr = AOSOLogitBoost_mex('train',...
          X,Y,var_cat_mask,...
        T, J, v,...
        node_size);
    end
    
    function [NumIter, TrLoss, F, P, Trees] = get (obj)
      [NumIter, TrLoss, F, P] = AOSOLogitBoost_mex('get', obj.ptr);
      Trees(NumIter,1) = struct('nodes',[],'splits',[],'leaves',[]);
      for i= 1:NumIter
          [Trees(i).nodes, Trees(i).splits, Trees(i).leaves] = ...
              AOSOLogitBoost_mex('save', obj.ptr, i);
          Trees(i).nodes = Trees(i).nodes + 1;
          Trees(i).splits(1, :) = Trees(i).splits(1, :) +1;
          Trees(i).leaves(1:2, :) = Trees(i).leaves(1:2, :) +1;
      end
    end
    
    function Y = predict(obj, X, T)
      if (nargin==2) 
        [T,~] = AOSOLogitBoost_mex('get',obj.ptr);
      end
      Y = AOSOLogitBoost_mex('predict', obj.ptr, X, T);
    end
    
    function delete(obj)
      AOSOLogitBoost_mex('delete',obj.ptr);
    end
  end % method
  
end % 

function [var_cat_mask, T, J, v, node_size] = parse_input(varargin)
  var_cat_mask = [];
  T = 5;
  J = 8;
  v = 1;
  node_size = 5;
  for i = 1 : 2 : nargin
    name = varargin{i};
    switch name
      case 'var_cat_mask'
        var_cat_mask = varargin{i+1};
      case 'T'
        T = varargin{i+1};
      case 'v'
        v = varargin{i+1};
      case 'J'
        J = varargin{i+1};
      case 'node_size'
        node_size  = varargin{i+1};
      otherwise
        error('Unknow properties');
    end % switch
  end % for
end


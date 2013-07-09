classdef batch_pAOSOLogitBoost < batch_boost_basic
  %BATCH_AOSOLOGITBOOST Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOLogitBoost';
    end
    
  end
  
end


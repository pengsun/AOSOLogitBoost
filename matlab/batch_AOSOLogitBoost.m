classdef batch_AOSOLogitBoost < batch_boost_basic
  %BATCH_AOSOLOGITBOOST Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = AOSOLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'AOSOLogitBoost';
    end
    
  end
  
end


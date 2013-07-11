classdef uci_VTLogitBoost < uci_boost_basic
  %UCI_AOSOBoostlog Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'VTLogitBoost';
    end
    
  end
  
end


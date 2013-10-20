classdef batch_pAOSOLogitBoostV2 < batch_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOLogitBoostV2();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOLogitBoostV2';
    end
    
  end % methods
  
end %


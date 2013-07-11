classdef uci_par_VTLogitBoost < uci_par_Boost
  %UCI_PAR_VTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Static)
    function run(fn_data,dir_rst, T,num_Tpre,v,J,ns)
      h = uci_VTLogitBoost(); %#ok<*SAGROW>
      h.num_Tpre = num_Tpre;
      h.T = T;
      h.cv = {v};
      h.cJ = {J};
      h.cns = {ns};
      run_all_param(h, fn_data, dir_rst);
      clear h;
    end
    
    function create_tasks(job, cdata,cv,cJ,cns)
      [~,paras] = uci_par_Boost.para_convert(cdata, cv, cJ, cns);
      for ix = 1 : numel(paras)
        fn_data = paras{ix}{1};
        dir_rst =  paras{ix}{2};
        T = paras{ix}{3};
        num_Tpre = paras{ix}{4};
        v = paras{ix}{5};
        J = paras{ix}{6};
        ns = paras{ix}{7};
        createTask(job, @uci_par_VTLogitBoost.run,...
          0, {fn_data,dir_rst, T,num_Tpre,v,J,ns});
        % log
        %
        str_dataset = uci_par_VTLogitBoost.dataset_to_str(fn_data);
        str_para = uci_par_VTLogitBoost.param_to_str(T,v,J,ns);
        str = [str_dataset,' ',str_para,' '];
        set(job.Tasks(ix), 'UserData',str);
      end
      
    end
    
    function print(job)
      str_algo = 'VTLogitBoost';
      fprintf(str_algo); fprintf('\n');
      uci_par_Boost.print_tasks(job);
    end % print
  end % method
  
end


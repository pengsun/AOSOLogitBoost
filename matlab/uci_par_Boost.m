classdef uci_par_Boost
    %UCI_PAR_BOOST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)

    function [paras,rand_paras] = para_convert(cfndir, cv, cJ, cns)
      paras = {};
      ndata = numel(cfndir);
      for ix_data = 1 : ndata
        fn_data = cfndir{ix_data}{1};
        dir_out = cfndir{ix_data}{2};
        T = cfndir{ix_data}{3};
        Tpre = cfndir{ix_data}{4};
        for ix1 = 1 : numel(cv)
          v = cv{ix1};
          for ix2 = 1 : numel(cJ)
            J = cJ{ix2};
            for ix3 = 1 : numel(cns)
              ns = cns{ix3};
              paras{end+1} = {fn_data,dir_out, T,Tpre, v,J,ns}; %#ok<AGROW>
            end
          end
        end
      end % for ix_data
      % random permutation for balanced load in distributed computation
      tmp_idx = randperm(numel(paras));
      rand_paras = paras(tmp_idx);
    end % para_convert
    function paras = param_to_singleloop(cv,cJ,cns)
      n = 1;
      for ix2 = 1 : numel(cv)
        obj.v = cv{ix2};
        for ix3 = 1 : numel(cJ)
          obj.J = cJ{ix3};
          for ix4 = 1 : numel(cns)
            obj.ns = cns{ix4};
            
            paras(n,1:3) = [cv{ix2}, cJ{ix3}, cns{ix4}]; %#ok<*AGROW>
            n = n + 1;
          end % ix4
        end % ix3
      end % ix2
    end %  param_to_singleloop
    
    function ret = param_to_str(T,v,J,ns)
      tmpl = [...
        'T%d','_',...
        'v%1.1d','_',...
        'J%d','_',...
        'ns%d'];
      ret = sprintf(tmpl,...
        T,v,J,ns);
    end % param_to_str
    
    function ret = dataset_to_str(fn_data)
      tmp = fileparts(fn_data);
      [~,name] = fileparts(tmp);
      ret = name;
    end
    
    function print_tasks(job)
%       str_algo = 'AOTOBoostSol2Sel2gain';
%       fprintf(str_algo); fprintf('\n');
      for i = 1 : numel(job.Tasks)
        str1 = get(job.Tasks(i),'UserData');
        tmp = get(job.Tasks(i),'ErrorMessage');
        if (~isempty(tmp))
          str2 = 'Error!';
        else
          str2 = get(job.Tasks(i),'State');
        end
        fprintf([str2,' ',str1,' ']);
        
        str_beg = get(job.Tasks(i),'StartTime');
        str_end = get(job.Tasks(i),'FinishTime');
        str = [str_beg,' ',str_end,' '];
        fprintf(str); fprintf('\n');
      end
    end % print
    end
    
end


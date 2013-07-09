classdef batch_boost_basic
  % Basic class for booster running on uci dataset
  %   Detailed explanation goes here
  
  properties
    num_Tpre = 4;
    T = 1;
    v = 1;
    J = 2;
    ns = 1;
    
    cv = {1};
    cJ = {2};
    cns = {1};
  end
  
  methods

    function [hboost, t_tr, num_it, abs_grad, F, P, Trees] = run_train (obj, fn_data)
      
      % print algo name
      algo_name = get_algo_name(obj);
      fprintf(algo_name); fprintf('\n');
      
      % print data name
      [~,name, ~] = fileparts(fn_data);
      fprintf(name); fprintf('\n');
      
      % print param
      tmp = obj.param_to_str(...
        obj.T, obj.v, obj.J, obj.ns);
      fprintf(tmp); fprintf('\n');
      
      % train
      S = load( fn_data, '-mat','Xtr','Ytr','cat_mask' );
      Xtr = S.Xtr;
      Ytr = S.Ytr;
      catmask = S.cat_mask;
      clear S;
      %
      hboost = get_handle(obj);
      fprintf('train start at %s\n', datestr(clock));
      tic
      hboost = train(hboost, Xtr,Ytr,...
        'var_cat_mask',catmask,...
        'T',obj.T, 'v',obj.v, 'J',obj.J,...
        'node_size',obj.ns);
      t_tr = toc;
      fprintf('train end at %s\n', datestr(clock));      
      drawnow;
      
      % get train result
      [num_it,abs_grad, F, P, Trees] = get(hboost);
    end
    
    function [err_it,it,t_te] = run_predict (obj, hboost, fn_data, num_it)   
      % test data
      S = load( fn_data, '-mat','Xte','Yte');
      Xte = S.Xte;
      Yte = S.Yte;
      clear S;
      
      % incremental predict
      it = linspace(1,num_it, min(obj.num_Tpre,num_it) );
      it = round(it);
      err_it = zeros(1, numel(it));
      fprintf('predict start at %s\n', datestr(clock));
      tic;
      for i = 1 : numel(it)
        Tpre = it(i);
        F = predict(hboost, Xte, Tpre);      
        % error rate
        [~,yy] = max(F);
        yy = yy - 1;
        err_it(i) = sum(yy~=Yte);
      end
      t_te = toc;
      fprintf('predict end at %s\n', datestr(clock));
      fprintf('num_it = %d, err = %d\n', it(end),err_it(end));
      drawnow;
    end
    
    function run_all_param (obj, fn_data, dir_rst)
      
      if ( ~exist(dir_rst,'dir') )
        mkdir(dir_rst);
      end
        
      for ix2 = 1 : numel(obj.cv)
        obj.v = obj.cv{ix2};
        for ix3 = 1 : numel(obj.cJ)
          obj.J = obj.cJ{ix3};
          for ix4 = 1 : numel(obj.cns)
            obj.ns = obj.cns{ix4};
       
            % train
            [hboost, time_tr, num_it, abs_grad, F, P,Trees]= run_train(obj, fn_data);
            
            % incremental predict           
            [err_it,it,time_te] = run_predict(obj, hboost, fn_data, num_it);
            err = err_it(end);
            
            % save result
            fnrst = obj.param_to_str(...
                obj.T, obj.v, obj.J, obj.ns);
            fn_rst = fullfile(dir_rst, [fnrst,'.mat']);            
            save(fn_rst,... 
              'err','num_it','abs_grad','F', 'P', ...
              'it','err_it','time_tr','time_te', 'Trees');
          
            % delete
            fprintf('\n');
            delete(hboost);
            
          end % ix4
        end % ix3
      end % ix2

      
      fprintf('done at %s\n', datestr(clock) );
      fprintf('----------------\n\n\n');
    end % run_all_param
    
    function h = get_handle(obj) %#ok<MANU>
      h = [];
    end
    
    function na = get_algo_name(obj)
      na = 'haha';
    end
    
  end % methods
  
  methods(Static)
    function ret = param_to_str(T,v,J,ns)
      tmpl = [...
        'T%d','_',...
        'v%1.1d','_',...
        'J%d','_',...
        'ns%d'];
      ret = sprintf(tmpl,...
        T,v,J,ns);
    end % param_to_str
    
    function ret = predinc_param_to_str(Tpre,v,J,ns)
     tmpl = [...
        'Tpre=%d',', ',...
        'v=%1.1d',', ',...
        'J=%d',', ',...
        'ns=%d'];
      ret = sprintf(tmpl,...
        Tpre,v,J,ns);
    end
  end % methods

end
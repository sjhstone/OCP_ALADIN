function oao = OCP_ALADIN_interface( ip, apar, acfg, misc )

    include = @(p) addpath(genpath(p));
    include('aladin_solver');
    include('math_helper');
    
    import casadi.*;
    
    %% AUX. VARIABLES
    x = cell(ip.subs.N, 1);
    y = cell(ip.subs.N, 1);
    
    % wastes space here
    A = [...
        zeros(        (ip.subs.N-2)*ip.var.x.n, ip.subs.subvardim        );...
        -1*eye(ip.var.x.n) zeros(ip.var.x.n, ip.subs.subvardim-ip.var.x.n);...
        zeros(ip.var.x.n, ip.subs.subvardim-ip.var.x.n)    eye(ip.var.x.n);...
        zeros(        (ip.subs.N-2)*ip.var.x.n, ip.subs.subvardim        ) ...
        ];
    
    misc.A = cell(ip.subs.N, 1);
    misc.cA = [];
    for subi=1:ip.subs.N
        AClipH = (ip.subs.N - subi)*ip.var.x.n + 1;
        Ai =  A( AClipH : AClipH+(ip.subs.N-1)*ip.var.x.n-1 , : );
        misc.A{subi} = Ai;
        misc.cA = horzcat(misc.cA, Ai);
    end
    clear A AClipH Ai
    
    subvar = cell(ip.subs.N, 1);

    xi_trj = cell(ip.subs.N, 1);
    xi_trj(:) = {nan(ip.subs.subvardim, acfg.MAX_ITER)};
    if isfield(acfg.ig, 'x')
        igx = acfg.ig.x;
    else
        error('No initial guess of x is present');
    end
    
    if isfield(acfg.ig, 'lambda')
        iglambda = acfg.ig.lambda;
    else
        iglambda = ones(ip.var.lambda.n, 1);
    end
    for subi = 1:ip.subs.N
        xi_trj{subi}(:,1) = acfg.ig.x;
    end
    
    lambda_trj = nan( ip.var.lambda.n , acfg.MAX_ITER );
    lambda_trj(:,1) = iglambda;
    
    %% ALADIN MAIN LOOP
    for ITER = 1:acfg.MAX_ITER
        
        %% DECOUPLED NLP AND gi, Ci, Hi
        for subi = 1:ip.subs.N
            %% SUBSYSTEM
            [subvar{subi}] = decoupled_nlp( ITER, ...
                xi_trj{subi}(:,ITER), ...
                lambda_trj(:,ITER),   ...
                subi, ip, apar.dnlp, acfg, misc...
            );
        end
        
        %% CHECK TERMINATION CONDITION
        misc.cCi = [];
        misc.cyi = [];
        misc.Hi = [];
        misc.gi = [];
        misc.cqpeCi = [];
        misc.cqpebi = [];
        for subi=1:ip.subs.N
            misc.cCi = blkdiag(misc.cCi, subvar{subi}.Ci);
            misc.cyi = vertcat(misc.cyi, subvar{subi}.yi);
            misc.Hi = blkdiag(misc.Hi, subvar{subi}.Hi);
            misc.gi = vertcat(misc.gi, subvar{subi}.gi);
            
            misc.cqpeCi = blkdiag(misc.cqpeCi, subvar{subi}.qpeCi);
            misc.cqpebi = vertcat(misc.cqpebi, subvar{subi}.qpebi);
        end
        % constraint enforcement check
        satisfy_cec = ...
            acfg.tol >= norm(misc.cA*misc.cyi, 1);
        % step delta check
        satisfy_sdc = true;
        for subi = 1:ip.subs.N
            step_delta = apar.dnlp.rho * norm(apar.dnlp.Sigma*...
                    subvar{subi}.yi - xi_trj{subi}(:,ITER), 1);
            satisfy_sdc = satisfy_sdc && (acfg.tol >= step_delta);
            if satisfy_sdc == false
                break
            end
        end
        if satisfy_cec && satisfy_sdc
            disp(['[OCP_ALADIN] ',...
                'ALADIN termination condition satisfied, EXIT' ]...
            );
            break
        end
        
        %% COUPLED QP
        qpout = coupled_qp(ITER, subvar, ip, apar.cqp, acfg, misc);
        
        %% ALPHA SEARCH
        alpha = [1;1;1];
        
        %% UPDATE X
        for subi = 1:ip.subs.N
            updated_x = xi_trj{subi}(:,ITER) + ...
                alpha(1) * ( subvar{subi}.yi - xi_trj{subi}(:,ITER) ) + ...
                alpha(2) * qpout.delta_y(1+(subi-1)*ip.subs.subvardim : subi*ip.subs.subvardim ,1);
            xi_trj{subi}(:,ITER+1) = updated_x;...
        end
        
        %% UPDATE LAMBDA
        lambda_trj(:,ITER+1) = lambda_trj(:,ITER) + ...
            alpha(3)*(qpout.lambda_QP(1:ip.var.lambda.n) - lambda_trj(:,ITER));
        
        %% PLOT
        state_var = [];
        for subi=1:ip.subs.N-1
            state_var = [state_var; xi_trj{subi}(1:end-ip.var.x.n,ITER+1)];
        end
        state_var = [state_var; xi_trj{subi}(:,ITER+1)];
        
        figure(1)
        for i=1:ip.var.x.n
            subplot(ip.var.x.n,1,i)
            plot(0:ip.subs.thrz:ip.subs.total_thrz, state_var(i:(ip.var.x.n+ip.var.u.n):end,1));
        end
        
        figure(2)
        for i=1:ip.var.u.n
            subplot(ip.var.u.n,1,i)
            plot(ip.subs.thrz:ip.subs.thrz:ip.subs.total_thrz, state_var(ip.var.x.n+i:(ip.var.x.n+ip.var.u.n):end,1));
        end
        
    end
    
    if ITER == acfg.MAX_ITER
        disp(['[OCP_ALADIN] ',...
                'Reaching MAX_ITER, ', acfg.MAX_ITER, ' , EXIT' ]...
            );
    end
    
    oao = struct(...
        'x', xi_trj,...
        'lambda', lambda_trj ...
    );
    
end


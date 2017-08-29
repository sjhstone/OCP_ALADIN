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
    clear AClipH Ai
    
    subvar = cell(ip.subs.N, 1);

    xi_trj = cell(ip.subs.N, 1);
    xi_trj(:) = {nan(ip.subs.subvardim, acfg.MAX_ITER)};
    for subi = 1:ip.subs.N
        xi_trj{subi}(:,1) = acfg.ig.x;
    end
    
    lambda_trj = zeros( ip.var.lambda.n , acfg.MAX_ITER );
    lambda_trj(:,1) = acfg.ig.lambda;
    
    %% ALADIN MAIN LOOP
    for ITER = 1:acfg.MAX_ITER
        
        %% DECOUPLED NLP AND gi, Ci, Hi
        for subi = 1:ip.subs.N
            %% SUBSYSTEM
            [subvar{subi}] = decoupled_nlp( ITER, ...
                xi_trj{subi}(:,ITER), ...
                lambda_trj(:,ITER),   ...
                subi, ip, apar.dnlp, misc...
            );
        end
        
        %% CHECK TERMINATION CONDITION
        misc.cCi = [];
        misc.cyi = [];
        misc.Hi = [];
        misc.gi = [];
        for subi=1:ip.subs.N
            misc.cCi = blkdiag(misc.cCi, subvar{subi}.Ci);
            misc.cyi = vertcat(misc.cyi, subvar{subi}.yi);
            misc.Hi = blkdiag(misc.Hi, subvar{subi}.Hi);
            misc.gi = vertcat(misc.gi, subvar{subi}.gi);
        end
        % constraint enforcement check
        satisfy_cec = ...
            acfg.tol >= sum(abs(misc.cA*misc.cyi));
        % step delta check
        satisfy_sdc = true;
        for subi = 1:ip.subs.N
            satisfy_sdc = satisfy_sdc && ...
                acfg.tol >= apar.dnlp.rho * sum(abs(apar.dnlp.Sigma*subvar{subi}.yi - xi_trj{subi}(:,ITER)));
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
        
    end
    
    oao = struct(...
        'x', xi_trj,...
        'lambda', lambda_trj ...
    );
    
end


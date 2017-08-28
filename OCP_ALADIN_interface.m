function [oao] = OCP_ALADIN_interface( ip, apar, acfg, misc )

    include = @(p) addpath(genpath(p));
    include('aladin_solver');
    include('math_helper');
    
    import casadi.*;
    
    %% ASSISTING VARIABLES
    
    x = cell(ip.subs.N, 1);
    y = cell(ip.subs.N, 1);
    
    % wastes space here
    A = [...
        zeros(        (ip.subs.N-2)*ip.var.x.n, ip.subs.subvardim        );...
        -1*eye(ip.var.x.n) zeros(ip.var.x.n, ip.subs.subvardim-ip.var.x.n);...
        zeros(ip.var.x.n, ip.subs.subvardim-ip.var.x.n)    eye(ip.var.x.n);...
        zeros(        (ip.subs.N-2)*ip.var.x.n, ip.subs.subvardim        ) ...
        ];
    
    misc.A = A;
    
    subvar = cell(ip.subs.N, 1);
    
    initial_xi = [ip.var.x.ss];
    for itvi = 1:ip.subs.m
        initial_xi = [initial_xi; ip.var.u.ss; ip.var.x.ss];
    end

    xi_trj = cell(ip.subs.N, 1);
    xi_trj(:) = {zeros(ip.subs.subvardim, acfg.MAX_ITER)};
    for subi = 1:ip.subs.N
        xi_trj{subi}(:,1) = initial_xi;
    end
    
    lambda_trj = zeros( ip.var.lambda.n , acfg.MAX_ITER );
    lambda_trj(:,1) = acfg.ig.lambda;
    
    figure(1)
    
    %% ALADIN MAIN LOOP
    for ITER = 1:acfg.MAX_ITER
        
        %% DECOUPLED NLP AND gi, Ci, Hi
        for subi = 1:ip.subs.N
            %% SUBSYSTEM
            [subvar{subi}] = decoupled_nlp(   ...
                xi_trj{subi}(:,ITER), ...
                lambda_trj(:,ITER),   ...
                subi, ip, apar.dnlp, misc);
        end
        %% CHECK TERMINATION CONDITION
        
        
        %% COUPLED QP
        qpout = coupled_qp(ITER, subvar, ip, apar, acfg, misc);
        
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
    end
    
    oao = struct(...
        'x', xi_trj,...
        'lambda', lambda_trj...
        );
    

end


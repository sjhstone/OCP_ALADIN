function [ cqo ] = coupled_qp( at_iter, subvar, ip, apar, acfg, misc )
    
    import casadi.*;
    
    w = {};
    x = [];
    
    for subi = 1:ip.subs.N
        del_y = MX.sym( ['delta_y' num2str(subi)] , [ip.subs.subvardim 1]);
        w = {w{:}, del_y};
        x = [ x  ; del_y];
    end
        
    switch acfg.cqp.solver
        case 'ipopt'
            disp(['[OCP_ALADIN] ', 'Solving coupled QP using ',...
                acfg.cqp.solver, newline,...
                '(@ iteration ', num2str(at_iter) ]);
            
            cCi = [];
            cyi = [];
            Hi = [];
            gi = [];
            Ai = [];
            for subi=1:ip.subs.N
                cCi = blkdiag(cCi, subvar{subi}.Ci);
                cyi = vertcat(cyi, subvar{subi}.yi);
                Hi = blkdiag(Hi, subvar{subi}.Hi);
                gi = vertcat(gi, subvar{subi}.gi);
                Ai = horzcat(Ai, subvar{subi}.Ai);
            end
            
            cA = Ai;
            cb = cA*cyi;
            
            cextA = [cA; cCi];
            cextb = [cb; zeros(size(cCi,1), 1)];
            
            g = {cextA*x + cextb};
            lbg = zeros(size(g{1}));
            ubg = zeros(size(g{1}));
            
            f = 1/2 * x'*Hi*x + gi'*x;
            ALADIN_f = f;
            
            optimization_problem = struct(...
                'f', ALADIN_f,...
                'x', vertcat(w{:}),...
                'g', vertcat(g{:})...
                );
            
            solver = nlpsol('solver', 'ipopt', optimization_problem);
            
            solution = solver(...
                'x0', zeros(size(x)),...
                'lbg', lbg,...
                'ubg', ubg);
            
            cqo = struct(...
                'delta_y'  , full(solution.x),...
                'lambda_QP', full(solution.lam_g)...
                );
            
            disp('HERE');
        case 'qpoases'
            error(['[OCP_ALADIN] ', 'NotImplementedException: ' acfg.cqp.solver]);  
        otherwise
            error(['[OCP_ALADIN] ', 'InvalidSolverSettingException: ' acfg.cqp.solver])
    end
end


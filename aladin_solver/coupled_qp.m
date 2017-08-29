function cqo = coupled_qp( at_iter, subvar, ip, par, acfg, misc )

    disp(['[OCP_ALADIN] ',...
        'OCP_ALADIN hasn''t implemented slackness variable' ]...
    );
    
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
                '(@ iteration ', num2str(at_iter), ')' ]);
            
            % QP constraint b,
            % $ \sum_i{A_i y_i} $
            qpcb = misc.cA*misc.cyi;
            
            cextA = [misc.cA; misc.cCi];
            cextb = [qpcb; zeros(size(misc.cCi,1), 1)];
            
            g = {cextA*x + cextb};
            lbg = zeros(size(g{1}));
            ubg = zeros(size(g{1}));
            
            f = 1/2 * x'*misc.Hi*x + misc.gi'*x;
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
                'ubg', ubg ...
            );
            
            cqo = struct(...
                'delta_y'  , full(solution.x),...
                'lambda_QP', full(solution.lam_g)...
            );
        case 'qpoases'
            error(['[OCP_ALADIN] ', 'NotImplementedException: ' acfg.cqp.solver]);  
        otherwise
            error(['[OCP_ALADIN] ', 'InvalidInputException: ' acfg.cqp.solver])
    end
    
end


function [ dno ] = decoupled_nlp( xi, lambda, subi, ip, par, misc )
    
    import casadi.*;
    
    disp(['[OCP_ALADIN] ', 'Solving decoupled NLP: subproblem ',...
        num2str(subi) ' / ' num2str(ip.subs.N)]);
    
    lbx = [];
    ubx = [];
    
    g = {};
    h = [];
    lbg = [];
    ubg = [];
    
    f = 0;
    
    x0 = MX.sym('x0', [ip.var.x.n, 1]);
    x = {x0};
    y = [x0];
    
    y_now = x0;
    
    if subi == 1    % first subproblem
        lbx = [ip.var.x.x0];
        ubx = [ip.var.x.x0];
    else
        lbx = [-Inf * ones(ip.var.x.n,1)];
        ubx = [ Inf * ones(ip.var.x.n,1)];
    end
    
    for itv = 1:ip.subs.m
        u_now = MX.sym( ['u' num2str(itv-1)], [ip.var.u.n, 1] );
        F_now = misc.integrator('x', y_now, 'u', u_now);
        
        x_end = F_now.X;
        f = f + F_now.Z;
        
        y_now = MX.sym( ['x' num2str( itv )], [ip.var.x.n, 1] );
        x = { x{:}, u_now, y_now };
        y = [  y  ; u_now; y_now ];
        lbx = [ lbx; ip.var.u.lb; ip.var.x.lb ];
        ubx = [ ubx; ip.var.u.ub; ip.var.x.ub ];
        
        g = { g{:}, x_end-y_now };
        h = [  h  ; x_end-y_now ];
        lbg = [lbg; zeros(ip.var.x.n, 1)];
        ubg = [ubg; zeros(ip.var.x.n, 1)];
    end
    
    if subi == ip.subs.N    % last subproblem
        tmnc = ip.goal.tmnc('x', y_now);
        f = f + tmnc.tmnc;
    end
    
    startat = (ip.subs.N - subi)*ip.var.x.n + 1;
    Ai = misc.A(...
        startat : startat+(ip.subs.N-1)*ip.var.x.n-1 ,...
        :);
    
    ALADIN_f = f + lambda' * Ai * y ...
                 + 1/2 * par.rho * (y-xi)' * par.Sigma * (y-xi);
    
    optimization_problem = struct(...
        'f', ALADIN_f,...
        'x', vertcat(x{:}),...
        'g', vertcat(g{:}) ...
        );
    
    solver = nlpsol('solver', 'ipopt', optimization_problem);
    
%     solution = solver(...
%         'x0', zeros(ip.subs.subvardim, 1), ...
%         'lbx', lbx, 'ubx', ubx,  ...
%         'lbg', lbg, 'ubg', ubg);
    xhat = [repmat([ip.var.x.ss; ip.var.u.ss], ip.subs.m, 1); ip.var.x.ss];
    solution = solver(...
        'x0', xhat, ...
        'lbx', lbx, 'ubx', ubx,  ...
        'lbg', lbg, 'ubg', ubg);    

    yi     = full(solution.x    );
    kappai = full(solution.lam_g);
    
    gi = gradient(f, y);
    gi = Function( ['g' num2str(subi)] , {y}, {gi}, {'x'}, {'g'});
    gi = gi('x', yi);
    gi = full(gi.g);
    
    Ci = jacobian(h, y);
    Ci = Function( ['C' num2str(subi)] , {y}, {Ci}, {'x'}, {'C'});
    Ci = Ci('x', yi);
    Ci = full(Ci.C);
    
    Hi = hessian( f + kappai'*h, y );
    Hi = Function('Hi', {y}, {Hi}, {'x'}, {'H'});
    Hi = Hi('x', yi);
    Hi = full(Hi.H);
    
    dno = struct(...
        'Ai'    , Ai,...
        'yi'    , yi,...
        'kappai', kappai,...
        'gi'    , gi,...
        'Ci'    , Ci,...
        'Hi'    , Hi);

end


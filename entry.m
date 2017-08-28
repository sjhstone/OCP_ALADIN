clc;
clear;

include = @(p) addpath(genpath(p));

include('class_def');
include('math_helper');
include('example');

import casadi.*;

varcfg.x.n  = 4;
% varcfg.x.x0 = zeros(4,1);
varcfg.x.lb = -Inf*ones(4,1);
varcfg.x.ub =  Inf*ones(4,1);
varcfg.x.ss = [2.1402105301746182e00 1.0903043613077321e00 1.1419108442079495e02 1.1290659291045561e02]';
varcfg.x.x0 = [2.1402105301746182e00+0.5*rand() 1.0903043613077321e00+0.5*rand() 1.1419108442079495e02+10*rand() 1.1290659291045561e02+10*rand()]';

varcfg.u.n  = 2;
varcfg.u.lb = [3; -9000];
varcfg.u.ub = [35;    0];
varcfg.u.ss = [14.19 -1113.50]';


%% EXTRA
x = MX.sym('x', [varcfg.x.n, 1]);
u = MX.sym('u', [varcfg.u.n, 1]);
Q = diag([0.2; 1.0; 0.5; 0.2]);
R = diag([0.5; 5e-7]);
zdot = (x - varcfg.x.ss)' * Q * (x - varcfg.x.ss) + (u - varcfg.u.ss)' * R * (u - varcfg.u.ss);
xdot = CSTR(x, u);
linearized_approx_A = jacobian(xdot, x);
linearized_approx_B = jacobian(xdot, u);
A_func = Function('A', {x, u}, {linearized_approx_A}, {'x', 'u'}, {'A'});
B_func = Function('B', {x, u}, {linearized_approx_B}, {'x', 'u'}, {'B'});
A = A_func('x', varcfg.x.ss, 'u', varcfg.u.ss);
A = full(A.A);
B = B_func('x', varcfg.x.ss, 'u', varcfg.u.ss);
B = full(B.B);
P = care(A, B, Q, R);

%% Basic Settings

subscfg.m     = 2;
subscfg.N     = 2;
subscfg.thrz  = 150;

f = Function('f', {x, u}, {xdot, zdot}, {'x', 'u'}, {'xdot', 'zdot'});
F = RK_Dormand_Prince(f, x, u, subscfg.thrz);

tmnc = (x-varcfg.x.ss)'*P*(x-varcfg.x.ss);
tmncf = Function('tmnc', {x}, {tmnc}, {'x'}, {'tmnc'});

goalcfg.stgc  = F;%.Z;
goalcfg.tmnc  = tmncf;

syscfg.type   = 'nonlinear';
syscfg.dx     = xdot;

%% Generate OCP instance
iOCP = OCP(varcfg, syscfg, goalcfg, subscfg);

%% ALADIN parameters
aladin_para.dnlp.rho = 1;
aladin_para.dnlp.Sigma = eye(subscfg.m * (varcfg.u.n+varcfg.x.n) + varcfg.x.n);

aladin_para.cqp.mu = [];

%% ALADIN configurations
aladin_cfg.ig.x       = [];
aladin_cfg.ig.lambda  = [...
-790.154020213032;...
-405.176780065470;...
-447.492152935987;...
-127.725599579678];
aladin_cfg.cqp.solver = 'ipopt';
aladin_cfg.tol        = eps;
aladin_cfg.MAX_ITER   = 20;

misc.integrator = F;


%% RUN ALADIN Interface
iOCP.printinfo_thrz();

aladin_result = OCP_ALADIN_interface(iOCP, aladin_para, aladin_cfg, misc);

vec_x = 

figure
for i=1:4
    subplot(4,1,i);
    hold on;
    plot(1:aladin_cfg.MAX_ITER, aladin_result(1).x(i,2:end));
    plot(1:aladin_cfg.MAX_ITER, aladin_result(1).x(6+i,2:end));
    plot(1:aladin_cfg.MAX_ITER, aladin_result(1).x(12+i,2:end));
    plot(1:aladin_cfg.MAX_ITER, aladin_result(2).x(6+i,2:end));
    plot(1:aladin_cfg.MAX_ITER, aladin_result(2).x(12+i,2:end));
    hold off;
end

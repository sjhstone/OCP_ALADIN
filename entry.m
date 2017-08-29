clc;
clf;
clear;

include = @(p) addpath(genpath(p));

include('class_def');
include('math_helper');
include('example');

import casadi.*;

PROBLEM_NAME = 'CSTR';

%% Variable Settings
% settings of OCP variables

% state at steady state
varcfg.x.ss = [2.1402105301746182e00 1.0903043613077321e00 1.1419108442079495e02 1.1290659291045561e02]';
% dimension of state
varcfg.x.n  = 4;
% initial state
varcfg.x.x0 = [1.0; 0.5; 100; 100];
% lower and upper bound of state
varcfg.x.lb = -Inf*ones(4,1);
varcfg.x.ub =  Inf*ones(4,1);

% control at steady state
varcfg.u.n  = 2;
% lower and upper bound of control
varcfg.u.lb = [3; -9000];
varcfg.u.ub = [35;    0];
% control at steady state
varcfg.u.ss = [14.19 -1113.50]';


%% Problem-Specific Settings
% Here, using CSTR system dynamic to create problem-specific integrator
% that incorporates stage cost and terminal cost
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

%% Subsystem Setting

% number of control intervals in one decoupled problem
subscfg.m     = 10;
% number of decoupled problems
subscfg.N     = 4;
% single control interval length ( m(k+1) - m(k) )
subscfg.thrz  = 50;

% generating terminal cost function
f = Function('f', {x, u}, {xdot, zdot}, {'x', 'u'}, {'xdot', 'zdot'});
F = RK_Dormand_Prince(f, x, u, subscfg.thrz);
tmnc = (x-varcfg.x.ss)'*P*(x-varcfg.x.ss);
tmncf = Function('tmnc', {x}, {tmnc}, {'x'}, {'tmnc'});

% TODO: stage cost
goalcfg.stgc  = F;
% terminal cost function
goalcfg.tmnc  = tmncf;

%% TODO: Dynamic System Configuration
% TODO: dynamic system type
syscfg.type   = 'nonlinear'; % or, 'linear'
syscfg.dx     = xdot;

%% Generate OCP instance
iOCP = OCP(varcfg, syscfg, goalcfg, subscfg);

%% ALADIN parameters
% parameter: Decoupled NLP: rho (penalty parameter)
aladin_para.dnlp.rho   = 1;
% parameter: Decoupled NLP: Sigma (Scaling Matrix)
% NOTE: not implemented to sub-system level
aladin_para.dnlp.Sigma = eye(subscfg.m * (varcfg.u.n+varcfg.x.n) + varcfg.x.n);
% TODO parameter: Coupled QP: mu
aladin_para.cqp.mu     = [];

%% ALADIN configurations
% initial guess of optimization variable
% NOTE: not implemented to sub-system level
aladin_cfg.ig.x       = [varcfg.x.x0; repmat([varcfg.u.ss; varcfg.x.ss], subscfg.m, 1)];
% initial guess of dual variable
% NOTE: not implemented to sub-system level
aladin_cfg.ig.lambda  = repmat([-790.154020213032; -405.176780065470; -447.492152935987; -127.725599579678],subscfg.N-1,1);
% solver for solving coupled QP
% NOTE: only 'ipopt' is supported at present
aladin_cfg.cqp.solver = 'ipopt';
% error tolerance of ALADIN termination condition
aladin_cfg.tol        = 1e-4;
% error tolerance of ipopt's dual variable
aladin_cfg.dualtol    = 1e-3;
% max accepted iteration count
aladin_cfg.MAX_ITER   = 35;

%% Miscellaneous Required Variables
% integrator function
misc.integrator = F;

%% Launch ALADIN Interface
% check time horizon information
iOCP.printinfo_thrz();

aladin_result = OCP_ALADIN_interface(iOCP, aladin_para, aladin_cfg, misc);
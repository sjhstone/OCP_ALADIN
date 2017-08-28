classdef OCP
    %OCP Summary of this class goes here
    %   Detailed explanation goes here
    properties (Access = private)
        has_steady_state = false;
    end
    
    properties
        var
        sys
        goal
        subs
    end
    

    methods
        function iOCP = OCP(var, sys, goal, subs)
            iOCP.var = var;
            iOCP.sys = sys;
            iOCP.goal = goal;
            iOCP.subs = subs;
            
            iOCP.subs.subvardim = subs.m * (var.x.n + var.u.n) + var.x.n;
            iOCP.var.lambda.n = (subs.N-1)*var.x.n;
        end
        
        function x0 = generate_initial_guess(object)
            if isempty(object.var.x.ss)
                disp('No steady state provided');
            end
        end
        
        function [] = printinfo_thrz(object)
            disp(['[OCP_ALADIN] ', 'Time Horizon', newline,...
                  'Decoupled into ', num2str(object.subs.N),' subproblems', newline, ...
                  'total: ',...
                  num2str(object.subs.m * object.subs.N * object.subs.thrz), ' s', newline, ...
                  ' each: ', num2str(object.subs.m * object.subs.thrz), ' s']);
        end
    end
    
end


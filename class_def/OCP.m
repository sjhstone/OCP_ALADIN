classdef OCP
    %OCP class that represents an optimal control problem
    
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
            iOCP.subs.total_thrz = subs.m * subs.N * subs.thrz;
            iOCP.var.lambda.n = (subs.N-1)*var.x.n;
        end
        
        function x0 = generate_initial_guess(object)
            if isempty(object.var.x.ss)
                disp('No steady state provided');
            end
        end
        
        function [] = printinfo_thrz(object)
            disp(['[OCP_ALADIN] ', 'Time Horizon Confirmation', newline,...
                  'Decoupled into ', num2str(object.subs.N),' subproblems', newline, ...
                  'total: ',...
                  num2str(object.subs.total_thrz), ' s', newline, ...
                  ' each: ', num2str(object.subs.m * object.subs.thrz), ' s']);
            disp(['Press ANY key to continue']);
            pause;
        end
    end
    
end


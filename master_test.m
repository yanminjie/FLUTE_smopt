function master_test( )
my_startup;
addpath(genpath(cd));
%MASTER_TEST Summary of this function goes here
%   Detailed explanation goes here
LB = [-60, -120, 0, -60, 8, -12, -12, -12, -0.2];
UB = [0, -80,    0.4, 0, 12, 12,  12,  12,    0];

% The optimal value so far -- used when we want to fix certain dimensions
x_opt = [-29.874, -118.45, 0.1024, -40.343, 11.555, 2.5959, -7.7745, 6.1968, -0.08708];
x_bound = [LB; UB];

% Decide the subset of values to be used
idx_active = [1, 2, 3, 4, 5, 9];
x_bound = x_bound(:, idx_active);

% Log transform
transform.fwd = @log;
transform.inverse = @exp;

% no transform
transform.fwd = @(x) x;
transform.inverse = @(x) x;

% [x_opt, y_opt, X, y] = surropt(@(x) log(sm_runSimulationFcn_parallel(x)), [], x_bound);
[x_opt, y_opt, X, y] = surropt(@(x) wrapped_fun(x, idx_active, x_opt), [], x_bound, transform);

file_name = sprintf('surropt_result_%s.mat', datestr(now, 30));
save(fullfile('results', file_name), 'X', 'y');

end

function y = wrapped_fun(x_sub, idx_active, x_opt)
% idx_acitve corresponds to the indices of the x_sub passed in

x_full = x_opt;
x_full(idx_active) = x_sub;

y = sm_runSimulationFcn_parallel(x_full);
end


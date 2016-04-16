function master_test( )
my_startup;
addpath(genpath(cd));
%MASTER_TEST Summary of this function goes here
%   Detailed explanation goes here
LB = [-60, -120, 0, -60, 8, -12, -12, -12, -0.2];
UB = [0, -80,    0.4, 0, 12, 12,  12,  12,    0];
% x_opt = [-23, -120, 0.145, -30.57, 10, 5.66, -10.78, 5.66, -0.1023];
x_bound = [LB; UB];
% y = sm_runSimulationFcn_parallel(x_opt);

[x_opt, y_opt, X, Y] = surropt(@(x) log(sm_runSimulationFcn_parallel(x)), [], x_bound);

file_name = sprintf('surropt_result_%s.mat', datestr(now, 30));
save(fullfile('results', file_name), 'X', 'y');

end


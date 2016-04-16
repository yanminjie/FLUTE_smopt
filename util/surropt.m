function [x_opt, y_opt, X, Y] = surropt(fun, x0, x_bound)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Test mode
if nargin == 0
    [fun, x0, x_bound] = get_dummy_fun();
end

%% Some parameters
N_init = 50;
N_per_batch = 4;
N_batch = 80;
N_dim = size(x_bound,2);
b_extreme = true; % Always deploy the next batch centred on the current overall optimum
b_opt_shrink = true; % Only allow surrogate model to give an optimal inside a box, not the whole range

% Test if parfor is avialable
try 
    % Testing parallel toolbox
    fprintf('Testing parallel toolbox \n')
    w = getCurrentWorker();
    fprintf('Parallel toolbox is available.\n')
    b_parallel = true;
catch ME
    b_parallel = false;
    fprintf('Parallel toolbox is unavailable with msg: %s .\n', ME.message)
end

%% Run the loop 
X = []; y = [];
x_opt = mean(x_bound);
i_iter = 1;

% Add plots for tracking progress
figure(101); clf; 
ha = axes(); hold on;
% h_yplot = plot(nan, nan);
set(ha, 'yscale', 'log');
h_title = title(' ');

% TODO: Add a title for displaying the optimum value

% And log file?

while i_iter <= N_batch
    % Design new batch of experiment
    if i_iter == 1 % Initial batch
        N = N_init;
        shrink = 1;
    else
        N = N_per_batch;
        shrink = 0.1;
    end
    [X_new, x_bound_batch] = design_new_batch(x_opt, N, x_bound, shrink);
    
    % Deploy the experiment
    [X_new, y_new] = deploy_experiment(fun, X_new, b_parallel); 
    
    % Concatenate the data
    X = [X; X_new]; y = [y; y_new];
     
    % Normalise data
    [X_norm, X_mean, X_std] = normalise_data(X);
    [y_norm, y_mean, y_std] = normalise_data(y);
    x_opt_norm = (x_opt - X_mean)./X_std;
    x_bound_norm = bsxfun(@rdivide, bsxfun(@minus, x_bound, X_mean), X_std);
    x_bound_batch_norm = bsxfun(@rdivide, bsxfun(@minus, x_bound_batch, X_mean), X_std);
    
    % First build a surrogate model
    timer = tic();
    hyp_opt = trainGP(X_norm, y_norm);
    % hypOpt = hyp_data.hyp(iOut);
    model_gp = assembleGP(hyp_opt);
    fprintf('Surrogate model building finished in %.1f seconds\n', toc(timer));
    
    % Then optimise the model to get the optimum
    % We can do multiple random starts since the surrogate model is cheap
    % to optimise!
    timer = tic();
    if b_opt_shrink && i_iter~=1
        lb = x_bound_batch_norm(1,:); ub = x_bound_batch_norm(2,:);
    else
        lb = x_bound_norm(1,:); ub = x_bound_norm(2,:);
    end
    options = optimoptions('fmincon','GradObj','on', 'Display', 'none');
    num_random_start = 100;
    y_random_start = inf; 
    for i_random = 1:num_random_start
        x_init = rand(1,N_dim).*(ub - lb) + lb;
        new_x_random_start = fmincon(@(x) opt_wrapper(x, model_gp), x_init, [], [], [], [], lb, ub, [], options);
        new_y_random_start = opt_wrapper(new_x_random_start, model_gp);
        if new_y_random_start < y_random_start
            y_random_start = new_y_random_start;
            x_opt_norm = new_x_random_start;
        end
    end
    fprintf('Surrogate model optimisation with %i random starts finished in %.1f seconds\n',...
                                                num_random_start, toc(timer));
    
    % Unormalise data
    x_opt = x_opt_norm.*X_std + X_mean;
    y_opt = feval(fun, x_opt);
    y_pred = model_gp.predictFun(x_opt_norm)*y_std + y_mean;
    y_std = model_gp.stdFun(x_opt_norm)*y_std;  
    fprintf('Prediction is  : %10.3e with 2std: %10.3e \n', exp(y_pred), exp(2*y_std))
    fprintf('Actual value is: %10.3e \n', exp(y_opt));
%     fprintf(['Solution is ', repmat('%10.3e', 1, length(x_opt)), '\n'], x_opt)
    
    % Only add to result if there is no nan
    if ~any(isnan([x_opt, y_opt]))
        X = [X; x_opt]; y = [y; y_opt];
    end
    
    % Update convergence plots
    % add model prediciton and std
    plot(ha, i_iter, exp(y_pred), 'bx')
    plot(ha, [i_iter, i_iter], exp([y_pred-2*y_std, y_pred+2*y_std]), 'b--')
    
    % add real value at model predicted x
    plot(ha, i_iter, exp(y_opt), 'rx');
    
    % add the rest of the batch value
    plot(ha, i_iter*ones(size(y_new)), exp(y_new), 'k+')
    
    % add historic minimum
    plot(ha, i_iter, exp(min(y)), 'ro');
    
    % update title display
    set(h_title, 'string', sprintf('Current best: %.4e', exp(min(y))));
    
    drawnow();
        
    % Decide where to deploy experiment next    
    if b_extreme
        % Or, more extreme, the next batch is always based on the best
        % historical point
        [~, idx] = min(y);
        x_opt = X(idx, :);       
    else
        % select where to deploy experiment next. If the GP model optimal point
        % is better than all others in the batch, use GP mode point, otherwise
        % use the best evaluated point.
        if min(y_new) < y_opt
            [y_new_real, idx] = min(y_new);
            x_opt = X_new(idx,:);
        end 
    end
        
    fprintf('Finished %2ith batch \n \n', i_iter)   
    i_iter = i_iter+1;  
end


end

%% 
function [fun, x0, x_bound] = get_dummy_fun()
N_dim = 5;
% Quadratic test function
scale = 10;
x_opt = 1:N_dim;
x_bound = repmat([-1; 1], 1, N_dim)*scale;
x0 = rand(1,N_dim)*scale;

fun = @(x) log(sum((x-x_opt).^2)+1);


end


function [X, x_bound_batch]= design_new_batch(x0, N, x_bound, shrink)
width = diff(x_bound)*shrink;
% if x0 is too close to the bound, move it away
x0 = max([x_bound(1,:) + width/2; x0]);
x0 = min([x_bound(2,:) - width/2; x0]);

% Record the new batch bound
x_bound_batch = [x0 - width/2;
                 x0 + width/2];
% Use LHS design to get normalised design points
X = lhsdesign(N, length(x0));

% Un-nofrmalise the design points
X = bsxfun(@plus, bsxfun(@times, X-0.5, width), x0);
end

% Deploy and clean up experiment result
function [X, y] = deploy_experiment(fun, X, b_parallel)
if nargin<3
    b_parallel = false;
end
N_exp = size(X,1);
y = nan(N_exp, 1);

if b_parallel && N_exp > 1
    parfor i_exp = 1:N_exp
        timer = tic();
        y(i_exp) = feval(fun, X(i_exp, :));
        fprintf('Experiment #%i/%i finished in %4.1f seconds \n', i_exp, N_exp, toc(timer));
    end
else
    for i_exp = 1:N_exp
        timer = tic();
        y(i_exp) = feval(fun, X(i_exp, :));
        fprintf('Experiment #%i/%i finished in %4.1f seconds \n', i_exp, N_exp, toc(timer));
    end
end

% idx_valid = ~isnan(y);
% y = y(idx_valid); X = X(idx_valid, :);

% penalise nan with large values
y(isnan(y)) = log(5e-12);
end

function [X_norm, X_mean, X_std] = normalise_data(X)
X_mean = mean(X);
X_std = std(X);
X_norm = bsxfun(@rdivide, bsxfun(@minus, X, X_mean), X_std);
end

% Add points to exiting plots
function append_plot(h_plot, x, y)
x_old = get(h_plot, 'XData');
y_old = get(h_plot, 'YData');
x = [x_old(:); x];
y = [y_old(:); y];

set(h_plot, 'XData', x, 'YData', y);

end

%% Functions for surrogate modelling
function [f, df] = opt_wrapper(x, model)
f = model.predictFun(x);
if nargout > 1
    df = model.gradFun(x);
end
end

function [f, fstd] = wrapper_func(modelGP, i_output, x, xmean, xstd, ymean, ystd)
x = bsxfun(@rdivide, bsxfun(@minus, x, xmean), xstd); 
f = modelGP(i_output).predictFun(x);
fstd = modelGP(i_output).stdFun(x);

f = f*ystd(i_output) + ymean(i_output);
fstd = fstd*ystd(i_output);

end

function modelGP = assembleGP(hyp)
modelGP.hyp = hyp; xTrain = hyp.xTrain; yTrain = hyp.yTrain;
yMean = mean(yTrain,1); yTrain = bsxfun(@minus,yTrain,yMean);

N = size(xTrain,1);
logell = hyp.cov.ell;
logsigmaf = hyp.cov.sigmaf; logsigman = hyp.lik;
sigman2 = exp(2*logsigman); SigmaLik = sigman2*eye(N);
jitter = 1*1e-6; hypCov = [logell;logsigmaf];
Kf = covSEard(hypCov,xTrain)+(exp(2*logsigmaf)*jitter)*eye(N);
Ky = Kf+SigmaLik; Ly = chol(Ky,'lower'); % Ly*Ly' = Ky;
invLy = Ly\eye(N); invKy = invLy'*invLy;
meanMult = invKy*yTrain;

modelGP.predictFun = @(x) predictGP(hypCov,meanMult,xTrain,x) + yMean;
modelGP.gradFun = @(x) gradGP(hypCov,meanMult,xTrain,x);
modelGP.stdFun = @(x) stdGP(hypCov,invLy,xTrain,x);
end

function f = predictGP(hypCov,meanMult,xTrain,xTest)
f = nan(size(xTest,1),size(meanMult,2));
idxNan = any(isnan(xTest),2);
xTest = xTest(~idxNan,:);
Ks = covSEard(hypCov,xTest,xTrain);
f(~idxNan,:) = Ks*meanMult;
end

function grad = gradGP(hypCov,meanMult,xTrain,xTest)
Ks = covSEard(hypCov,xTest,xTrain);
nIn = size(xTrain,2);
nOut = size(meanMult,2);
grad = zeros(nOut,nIn);
ell = exp(hypCov(1:nIn));
for iDim = 1:nIn
    grad(:,iDim) = (Ks.*(-dist(xTest(iDim),xTrain(:,iDim))/ell(iDim)^2))*meanMult;
end
end

function std = stdGP(hypCov,invLy,xTrain,xTest)
Ks = covSEard(hypCov,xTrain,xTest);
invLyKs = invLy*Ks;
logsigmaf = hypCov(end);
std = sqrt(exp(logsigmaf*2)-sum(invLyKs.*invLyKs,1)');
end

function [xn, xmean, xstd ]= normalise(x)
xmean = mean(x);
xstd = std(x);
xn = bsxfun(@rdivide, bsxfun(@minus, x, xmean), xstd);
end


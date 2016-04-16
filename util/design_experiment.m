function X = design_experiment(N, range)
% design_experiment designs points to perform experiments using Latin
% Hypercube sampling.
%
% Inputs:
%       N:      number of experiments (rows of X)
%   range:      a Dx2 matrix containing the range of each dimension. The
%               left column denotes the lower-bound while the right column
%               denotes the upper bound. D is the dimension of the input.
%
% Output:
%       X:      A design matrix containing all the experiment inputs. The
%               dimension of the matrix is NxD


% Pre-processing
D = size(range, 1); 
delta_range = range(:,2) - range(:,1);

% Use LHS design to get normalised design points
X = lhsdesign(N, D);

% Un-nofrmalise the design points
X = bsxfun(@plus, bsxfun(@times, X, delta_range'), range(:,1)');

end
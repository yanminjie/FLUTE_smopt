function hypOpt = trainGP(X,Y)
xRange = (max(X)-min(X))'; 
yMean = mean(Y); Y = bsxfun(@minus, Y, yMean);
hypStart.cov.sigmaf = log(std(Y)/2);
hypStart.cov.ell = log(xRange/2);
hypStart.lik = log(std(Y)/4);

[hypOpt, fVal] = minimize(hypStart,@fullGPNegLogLik, -500, X, Y);
hypOpt.xTrain = X; hypOpt.yTrain = bsxfun(@plus, Y, yMean);
end

function [ nlZ, dnlZ ] = fullGPNegLogLik(hyp,X,Y)
%FULLGPNEGLOGLIK Summary of this function goes here
%   Detailed explanation goes here

logsigman = hyp.lik; logell = hyp.cov.ell; logsigmaf = hyp.cov.sigmaf;
N = size(X,1); Din = size(X,2); Dout = size(Y,2);
sigman = exp(2*logsigman); SigmaLik = sigman*eye(N); % noise matrix

jitter = 1*1e-6; hypCov = [logell;logsigmaf]; % reformat for covSEard
Kf = covSEard(hypCov,X)+(exp(2*logsigmaf)*jitter)*eye(N); 
Ky = Kf+SigmaLik; Ly = chol(Ky,'lower'); % Ly*Ly' = Ky;
invLy = Ly\eye(N); invKy = invLy'*invLy;
alpha = invKy*Y;

nlZ = 0.5*sum(sum(Y'*alpha)) + sum(log(diag(Ly))) + Dout*N/2*log(2*pi);

%% Start of derivative calculations
if nargout > 1
    % Pre-computation
    H = Dout*invKy-alpha*alpha';
    invEll2 = exp(-2*logell);
    % Derivative w.r.t. sigman
    dKy_sigman = 2*SigmaLik;
    dnlZ.lik = 1/2*sum(sum(H.*dKy_sigman));
    % Derivative w.r.t sigmaf
    dKy_sigmaf = 2*Ky-2*SigmaLik;
    dnlZ.cov.sigmaf = 1/2*sum(sum(H.*dKy_sigmaf)); 
    % Derivative w.r.t lengthscales
    dnlZ.cov.ell = zeros(Din,1);
    for jDim = 1:Din
        dKy_dlj = invEll2(jDim)*(Ky.*sq_dist(X(:,jDim)'));
        dnlZ.cov.ell(jDim) = 1/2*sum(sum(H.*dKy_dlj));
    end
end


end


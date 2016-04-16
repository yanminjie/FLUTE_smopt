function hyp = buildGPModels
%BUILDGPMODELS Summary of this function goes here
%   Detailed explanation goes here
addpath('util');
rng(870907);
data = load('testCombinationR1R2R3_YXuRequest20140630.mat');
X = data.I_inputs;
Y = [data.r1,data.r2,data.r3,data.mu];
nTrain = 256;
[nData,nOut] = size(Y);
idxTrain = randperm(nData,nTrain);
idxTest = setdiff(1:nData,idxTrain);

xTrain = X(idxTrain,:); yTrain = Y(idxTrain,:);
xTest = X(idxTest,:); yTest = Y(idxTest,:);

rmsError = zeros(2,nOut);
for iOut = 1:nOut
% hyp(iOut) = trainGP(xTrain,RTrain(:,iOut));
hypOpt = trainGP(X,Y(:,iOut));
hyp(iOut) = hypOpt;
modelGP(iOut) = assembleGP(hypOpt);
% training error
yPred = modelGP(iOut).predictFun(xTrain);
error = yTrain(:,iOut)-yPred; rmsError(1,iOut) = std(error);
% test error
yPred = modelGP(iOut).predictFun(xTest);
error = yTest(:,iOut)-yPred; rmsError(2,iOut) = std(error);
end

save '20140630Hyp.mat' hyp
end

%% functions for building GP models
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
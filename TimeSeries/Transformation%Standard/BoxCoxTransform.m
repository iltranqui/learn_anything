function yM = BoxCoxTransform(xV,lambdaV)
% yV = BoxCoxTransform(xV,lambda)
% BOXCOXTRANSFORM takes a time series 'xV' and applies the Box Cox power
% transform for given parameter lambda in 'lambdaV'.
% INPUTS:
% - xV      : vector of a scalar time series
% - lambdaV : vector of lambda values (can be a single lambda as well)
% OUTPUTS:
% - yM      : a matrix of the Box-Cox transformed time series.
xV = xV(:);
n = length(xV);
if isempty(lambdaV) 
    yM=[];
else
    nlambda = length(lambdaV);
    yM = NaN*ones(n,nlambda);
    for ilambda=1:nlambda
        lambda=lambdaV(ilambda);
        if lambda==0
            xmin = min(xV);
            if xmin<=0
                xV = xV-xmin+1;
            end
            yM(:,ilambda) = log(xV);
        else
            yM(:,ilambda) = (xV.^lambda-1)/lambda;
        end
    end
end

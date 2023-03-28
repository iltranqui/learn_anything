function yV = NormalizedTransform(xV)
    % xM = NormalizedTransform(xV)
    % NORMALIZEDTRANSFORM takes a time series 'xV' and transform it linearly to
    % a time series with zero mean and one standard deviation. 
    % INPUTS:
    % - xV      : vector of a scalar time series
    % OUTPUTS:
    % - yV      : a vector of equal size.
xV = xV(:);
mx = mean(xV);
xsd = std(xV);
yV = (xV - mx) / xsd;
end
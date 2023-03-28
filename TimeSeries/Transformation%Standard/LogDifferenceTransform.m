function yV = LogDifferenceTransform(xV)
    % yV = LogDifferenceTransform(xV)
    % LOGDIFFERENCETRANSFORM takes a time series 'xV' and computes the first 
    % differences of the logarithmically transformed time series. If 'xV'
    % contains negative values, first 'xV' is translated so that the smallest
    % value is one. 
    % INPUTS:
    % - xV      : vector of a scalar time series
    % OUTPUTS:
    % - yV      : a vector of equal size.
xV = xV(:);
xmin = min(xV);
if xmin<=0
    xV = xV-xmin+1;
end
yV = log(xV(2:end))-log(xV(1:end-1));

end
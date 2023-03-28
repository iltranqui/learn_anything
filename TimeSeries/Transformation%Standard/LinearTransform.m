function yV = LinearTransform(xV)
    % xM = LinearTransform(xV)
    % LINEARTRANSFORM takes a time series 'xV' and transform it linearly to
    % the interval [0,1] (min of 'xV' is 0 and max is 1). 
    % INPUTS:
    % - xV      : vector of a scalar time series
    % OUTPUTS:
    % - yV      : a vector of equal size.
    xV = xV(:);
    xmin = min(xV);
    xmax = max(xV);
    d = xmax - xmin;
    yV = (xV - xmin) / d;
end
    
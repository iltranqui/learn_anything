function yV = UniformTransform(xV)
    % xM = UniformTransform(xV)
    % UNIFORMTRANSFORM takes a time series 'xV' and changes its marginal
    % cumulative function to Uniform in [0,1]. 
    % INPUTS:
    % - xV      : vector of a scalar time series
    % OUTPUTS:
    % - yV      : a vector of equal size.

    xV = xV(:);
    n = length(xV);
    FxV = ([1:n]'-0.326)/(n+0.348); % The position plotting transform
    [tmpV,ixV] = sort(xV);
    [tmpV,jxV]=sort(ixV);
    yV = FxV(jxV);

end
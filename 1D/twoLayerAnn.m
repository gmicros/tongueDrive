function  [Wkj, Wji, y] = twoLayerAnn(features, targets, noHidden, iter)
    
    i = size(features, 1);
    j = noHidden;
    k = size(targets, 1);
    obs = size(features,2);
    alpha = 0.1;
    %% normalize inputs, initialize weights
    x = [zscore(features); 0.9*ones(1, obs)];
    d = 0.99*(2*targets - 1);

    Wji = rand(j, i+1) - 0.5;
    Wkj = rand(k, j+1) - 0.5;

    %% activation funk
    phi = @(x)tanh(x);
    phiPrimes = @(x)(1 - phi(x).^2);
    % use this if you haven't thrown away tanh(x)
    dev = @(x)(1 - x.^2);

    %% doiint
    for iter = 1:iter
       %% feed forward
       h = [ phi(Wji * x ); 0.9*ones(1, obs) ];
       y = phi(Wkj * h);

       %% backprop
       err = (d - y);

       delK = alpha * err .* dev(y);
       delJ = (delK' * Wkj)' .* dev(h);

       deltaK = delK * h'; 
       deltaJ = delJ * x';

       Wkj = Wkj + deltaK./obs;
       Wji = Wji + deltaJ(1:j, :)./obs;   

       er(iter) = mean(mean(err.^2));
    end
end
function [P, Z, S] = LRREKM(X, alpha, beta, gamma, epsilon, maxIter)
    
    lossMat = [];
    iterMat = [];
    

    rho = 1.01;
    mu = 0.1;
    maxMu = 10^5;
    [n, d] = size(X);
    L = getL(X');
    P = rand(d,d);
    Z = rand(n,n);
    C = zeros(n,n);
    S = getS(Z, C, alpha, mu);
    loss = getLoss(X, P, Z, L, alpha, beta, gamma);
    lossMat = [lossMat ; loss];
    iter = 0;
    iterMat = [iterMat ; iter];
    while iter <= maxIter
        loss0 = loss;
        Q = diag(0.5 ./ sqrt(sum(P .* P , 2) + eps));
        P = getP(X, Z, Q, L, P, beta, gamma);
        
        Z = getZ(X, P ,C, S, mu);
        S = getS(Z, C, alpha, mu);
        
        loss = getLoss(X, P, Z, L, alpha, beta, gamma);
        lossMat = [lossMat ; loss];
        if ((loss0-loss)'*(loss0-loss) < epsilon) || (loss > loss0)
            break;
        end 
        C = getC(C, Z, S, mu);
        mu = min(rho*mu, maxMu);
        iter = iter + 1;
        iterMat = [iterMat ; iter];
    end
    
end
function Z = getZ(X, P ,C, S, mu)
    [n, d] = size(X);
    Z = (X*P'*X'-(1/mu)*C-S)*pinv(X*P*P'*X'-eye(n));
end
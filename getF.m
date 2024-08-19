function F = getF(X, P, L, alpha)
    dim = size(L, 2);
    F = 2*pinv(2*eye(dim) + alpha*L + alpha*L')*X*P;
end
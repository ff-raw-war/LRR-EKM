function P = getP(X, Z, Q, L, P0, beta, gamma)
    P = pinv(gamma*X'*L*X+X'*Z'*Z*X+beta*Q)*X'*Z'*X;
end
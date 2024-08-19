function loss = getLoss(X, P, Z, L, alpha, beta, gamma)
    l21fsP = 0; % L21norm
    dimP = size(P , 2);
    for i = 1 : dimP
        l21fsP = l21fsP + norm(P( : , i), 2); 
    end
    [~, SigmaZ, ~] = svd(Z);
    loss = norm(X-Z*X*P,'fro')^2+alpha*trace(SigmaZ)+beta*l21fsP+gamma*trace(P'*X'*L*X*P);
end
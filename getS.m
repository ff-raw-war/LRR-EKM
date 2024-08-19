function S = getS(Z, C, alpha, mu)
    es = alpha/mu;
    temp_S = Z+C/mu;
    [uu,ss,vv] = svd(temp_S,'econ');
    ss = diag(ss);
    SVP = length(find(ss>es));
    if SVP>1
        ss = ss(1:SVP)-es;
    else
        SVP = 1;
        ss = 0;
    end
    S = uu(:,1:SVP)*diag(ss)*vv(:,1:SVP)';
end
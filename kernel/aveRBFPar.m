function par=aveRBFPar(data , size) 
    % 获取RBF核的参数delta，
    % 参数data是两个类混在一起的数据80*4， size是80
    mat_temp = sum(data.^2,2) * ones(1,size) + ones(size,1)*sum(data.^2,2)' - 2* data*data';
    % 就是data的平方。(data^2 + 2 data dataT + data ^2)
    tempMean = (1/size^2) * sum(sum(mat_temp,1),2) ;
    % sum(a,1)按列求和，（，2）是按行求和。1/80*80 * 所有元素的和
    par = sqrt(tempMean) ;
end
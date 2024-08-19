function [empKernelTrainData, empKernelTestData] = empKernelGenerator(trainData, testData, kernelType)
    % 通过经验核映射，将样本点映射到特征空间
    % 输出核矩阵
    % kernelType 可以是 'rbf' 等
    % 输入的data1、data2是n✖d的
    
    totalView = 1; % 视角数，多核核数

    tempKernelPar = aveRBFPar(trainData, size(trainData, 1)) ;
    % 计算出两个类串在一起的核参数，上一个例子MEKL是指定的6.907
    kdelta = 2^(-3)+(2^3 - 2^(-3))*rand(totalView,1) ;
    kernelPar = kdelta .* tempKernelPar;
    % 以上这段代码得到核参数kernelPar

    implicitKernelMat = Kernel(trainData , trainData , kernelType , kernelPar) ;
    [pc , variances , explained] = pcacov(implicitKernelMat);
    %[PC, LATENT, EXPLAINED] = pcacov(X) 中pc,latent,explained特征向量,特征值,比重
    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 %特征值大于等于10^(-3)，latent,explained都是由高到低拍好顺序的。
        if i+1 > size(variances,1) %如果超过特征值的个数
            label = 1 ;
            break ;
        end
        i = i + 1 ;    
    end

    if label == 0 
        i = i - 1 ;
    end

    index = 1 : i ;
    P = pc(: , index) ;%满足特征值条件的特征向量
    R = diag(variances(index)) ;%X = diag(v,k)以向量v的元素作为矩阵X的第k条对角线元素，当k=0时，v为X的主对角线；当k>0时，v为上方第k条对角线；当k<0时，v为下方第k条对角线。
    empKernelTrainData = implicitKernelMat * P * R^(-1/2) ;%相当于论文中的B=KAQ.^(-1/2)   

    kerTestMat = Kernel(testData , trainData , kernelType, kernelPar) ;%核化的测试矩阵，需要将测试集放入训练集一起算的
    empKernelTestData = kerTestMat * P * R^(-1/2) ;  
end
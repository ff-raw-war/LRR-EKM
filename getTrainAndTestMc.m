function [trainSet , testSet , labelMatTrain, labelMatTest, labelVecTrain, labelVecTest] = getTrainAndTestMc(dataSet , ithCycle , totalCycle)
    % 实现将原始dataSet按照交叉验证，融合成一个trainSet和一个testSet
    % 将第ithCycle行的几个segment加上label拼成testSet
    % 将其余行的几个segment拼成trainSet
    % 参数：
    % dataSet：segmentNum × totalClass个cell，每个cell是segSize × dim
    % ithCycle ：第i轮交叉
    % totalCycle：总轮数
    % 返回：
    % testSet：融合成一个集，不包含label列
    % trainSet：除去testSet的dataSet，不包含label
    % labelMat：是标签矩阵，n×c的。如第i行是0 0 1 0，说明第i个样本是第3类
    % labelVec：是原本的标签向量。
    
    totalClass = size(dataSet , 2);   
    trainSet = [];
    testSet = [];
    labelMatTrain = [];
    labelMatTest = [];
    labelVecTrain = [];
    labelVecTest = [];
    trainCells = cell(1 , totalClass); % 将4×3个cell合并为1×3个cell
    trainLabelCells = cell(1 , totalClass); % 保存三类标签到三个cell
    testCells = cell(1 , totalClass);
    testLabelCells = cell(1 , totalClass);
    
    lenTrain = 0; % 
    widTrain = 0;
    lenTest = 0;
    widTest = 0;

    % 将4×3个cell合并为1×3个cell
    for i = 1 : totalCycle
        for j = 1 : totalClass
            if i ~= ithCycle % 不是当前这一行的cell
                lenTrain = lenTrain + size(dataSet{i , j} , 1);
                widTrain = widTrain + size(dataSet{i , j} , 2);
                trainCells{1 , j} = [trainCells{1 , j} ; dataSet{i , j}(: , :)];
                trainLabelCells{1 , j}= [trainLabelCells{1 , j} ; ones(size(dataSet{i , j} , 1) , 1)];
            else
                lenTest = lenTest + size(dataSet{i , j} , 1);
                widTest = widTest + size(dataSet{i , j} , 2);
                testCells{1 , j} = [testCells{1 , j} ; dataSet{i , j}(: , :)];
                testLabelCells{1 , j}= [testLabelCells{1 , j} ; ones(size(dataSet{i , j} , 1) , 1)];
            end
        end
    end

    for i = 1 : totalClass % 得到label阵
        labelMatTrain = blkdiag(labelMatTrain , trainLabelCells{1 , i}); % 对角对接得到
        labelVecTrain = [labelVecTrain ; trainLabelCells{1, i}*i]; % 把label向量上下相接
        labelMatTest = blkdiag(labelMatTest , testLabelCells{1 , i});
        labelVecTest = [labelVecTest ; testLabelCells{1, i}*i];
        trainSet = [trainSet ; trainCells{1 , i}];
        testSet = [testSet ; testCells{1 , i}];
    end
    
    
end
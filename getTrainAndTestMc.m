function [trainSet , testSet , labelMatTrain, labelMatTest, labelVecTrain, labelVecTest] = getTrainAndTestMc(dataSet , ithCycle , totalCycle)
    % ʵ�ֽ�ԭʼdataSet���ս�����֤���ںϳ�һ��trainSet��һ��testSet
    % ����ithCycle�еļ���segment����labelƴ��testSet
    % �������еļ���segmentƴ��trainSet
    % ������
    % dataSet��segmentNum �� totalClass��cell��ÿ��cell��segSize �� dim
    % ithCycle ����i�ֽ���
    % totalCycle��������
    % ���أ�
    % testSet���ںϳ�һ������������label��
    % trainSet����ȥtestSet��dataSet��������label
    % labelMat���Ǳ�ǩ����n��c�ġ����i����0 0 1 0��˵����i�������ǵ�3��
    % labelVec����ԭ���ı�ǩ������
    
    totalClass = size(dataSet , 2);   
    trainSet = [];
    testSet = [];
    labelMatTrain = [];
    labelMatTest = [];
    labelVecTrain = [];
    labelVecTest = [];
    trainCells = cell(1 , totalClass); % ��4��3��cell�ϲ�Ϊ1��3��cell
    trainLabelCells = cell(1 , totalClass); % ���������ǩ������cell
    testCells = cell(1 , totalClass);
    testLabelCells = cell(1 , totalClass);
    
    lenTrain = 0; % 
    widTrain = 0;
    lenTest = 0;
    widTest = 0;

    % ��4��3��cell�ϲ�Ϊ1��3��cell
    for i = 1 : totalCycle
        for j = 1 : totalClass
            if i ~= ithCycle % ���ǵ�ǰ��һ�е�cell
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

    for i = 1 : totalClass % �õ�label��
        labelMatTrain = blkdiag(labelMatTrain , trainLabelCells{1 , i}); % �ԽǶԽӵõ�
        labelVecTrain = [labelVecTrain ; trainLabelCells{1, i}*i]; % ��label�����������
        labelMatTest = blkdiag(labelMatTest , testLabelCells{1 , i});
        labelVecTest = [labelVecTest ; testLabelCells{1, i}*i];
        trainSet = [trainSet ; trainCells{1 , i}];
        testSet = [testSet ; testCells{1 , i}];
    end
    
    
end
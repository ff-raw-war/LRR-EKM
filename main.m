clc;
clear all, close all;
start0 = tic;
addpath('./eval/');
addpath('./kernel/');
addpath('../../data/');

para = [10^-6; 10^-5; 10^-4; 10^-3; 10^-2; 10^-1 ; 10^0; 10^1; 10^2; 10^3; 10^4; 10^5; 10^6];
epsilon = 10^-3;
maxIter = 100;

dataSetName = 'CMU-PIE';
load(dataSetName);
method = 'LRR-EKM';  

X = fea;
label = gnd; 
dataSet = [X, label];
totalCycle = 5; % 5-fold Cross-Validation
[dataSet , dim] = dataSet2Segments(dataSet , totalCycle);
totalClass = numel(unique(label));

warning off;
fprintf('Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);
string = '\n';
fprintf(string);

result = [];

[N,D] = size(X);
minCE = 100;
bestAcc = 0;
bestNmi = 0;
bestPurity = 0;
bestTimePerTrain = 100000;


paraString = 'alpha:%.6f, beta:%.6f, gamma:%.6f';

for ithAlpha = 1 : size(para)
    for ithBeta = 1 : size(para)
        for ithGamma = 1 : size(para)
                        alpha = para(ithAlpha);
                        beta = para(ithBeta);
                        gamma = para(ithGamma);
    
                        paraCurrent = [alpha, beta, gamma];
    
                        fprintf(strcat('Current Parameters:',paraString,' \n'), paraCurrent);
    
                        for ithCycle = 1 : totalCycle 
                            
                            [trainSet , testSet , labelMatTrain, labelMatTest, labelVecTrain, labelVecTest] = getTrainAndTestMc(dataSet , ithCycle , totalCycle);
                            [Y, Ytest] = empKernelGenerator(trainSet, testSet, 'rbf');
                            Y = trainSet;
                            Ytest = testSet;

                            start1 = tic;
                            [P, Z, S] = LRREKM(Y, alpha, beta, gamma, epsilon, maxIter);
   
                            timePerTrain = toc(start1);
                            
                            Y = gather(Y);
                            P = gather(P);
                            mdl = fitcknn(Y*P,labelVecTrain,'NumNeighbors',1);
                            [labelTestPredict] = predict(mdl, Ytest*P);
    
                            [res] = my_eval_y1(labelTestPredict, labelVecTest); 
                            result = [result; [res(2) res(3) res(4) res(9) res(5) res(11) timePerTrain]]; 
                        end
                        result(totalCycle+1, :) = mean(result);  
                        result(totalCycle+2, :) = std(result(1:totalCycle, :)) ;
                        result(totalCycle+3, :) = max(result); 
     
                        avgAcc = result(totalCycle+1 , 1); 
                        avgNmi = result(totalCycle+1 , 2);
                        avgPurity = result(totalCycle+1 , 3);
                        avgF1 = result(totalCycle+1 , 4);
                        avgAR = result(totalCycle+1 , 5);
                        avgRecall = result(totalCycle+1, 6);
                        avgTimePerTrain = result(totalCycle+1, 7);
    
                        stdAcc = result(totalCycle+2 , 1);
                        stdNmi = result(totalCycle+2 , 2);
                        stdPurity = result(totalCycle+2 , 3);
                        stdF1 = result(totalCycle+2 , 4);
                        stdAR = result(totalCycle+2 , 5);
                        stdRecall = result(totalCycle+2, 6);
                        stdTimePerTrain = result(totalCycle+2, 7);
                        
                        if avgAcc > bestAcc
                            paraBest = paraCurrent;
                            fprintf(strcat('acc:%.4f±%.4f ','  \n'),avgAcc, stdAcc);
                            fprintf(strcat('nmi:%.4f±%.4f ','  \n'),avgNmi, stdNmi);
                            fprintf(strcat('purity:%.4f±%.4f ','  \n'),avgPurity, stdPurity);
                            fprintf(strcat('F1:%.4f±%.4f ','  \n'),avgF1, stdF1);
                            fprintf(strcat('AR:%.4f±%.4f ','  \n'),avgAR, stdAR);
                            fprintf(strcat('Recall:%.4f±%.4f ','  \n'),avgRecall, stdRecall);
                            fprintf(strcat('TimePerTrain:%.4f±%.4f ','  \n'),avgTimePerTrain, stdTimePerTrain);
    
                            bestAcc = avgAcc;
                            bestStdAcc = stdAcc;
                            bestNmi = avgNmi;
                            bestStdNmi = stdNmi;
                            bestPurity = avgPurity;
                            bestStdPurity = stdPurity;
                            bestF1 = avgF1;
                            bestStdF1 = stdF1;
                            bestAR = avgAR;
                            bestStdAR = stdAR;
                            bestRecall = avgRecall;
                            bestStdRecall = stdRecall;
                            bestTimePerTrain = avgTimePerTrain;
                            bestStdTimePerTrain = stdTimePerTrain;
                            
                            bestZ = Z;
                            bestP = P;

                            bestPredict = labelTestPredict;
                            %}
                        end 
        end
            
    end
end

fprintf('\n---------------------------------------------------------\n');
fprintf(strcat('acc:%.4f±%.4f \n nmi:%.4f±%.4f \n purity:%.4f±%.4f \n F1:%.4f±%.4f \n AR:%.4f±%.4f \n Recall:%.4f±%.4f \n timeTrain:%.4f±%.4f \n',paraString,'  \n'),bestAcc,bestStdAcc,bestNmi,bestStdNmi,bestPurity,bestStdPurity,bestF1,bestStdF1,bestAR,bestStdAR,bestRecall,bestStdRecall,bestTimePerTrain,bestStdTimePerTrain,paraBest);


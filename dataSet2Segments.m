function [dataSegments , dim] = dataSet2Segments(data , segmentNum)
    % �˺�����ԭʼ���ݼ����ֳ�Ƭ�β������������Ϊ��ʵ�ֺ���Ľ�����֤
    % ����:
    % data����sample���� �� feature���������һ��Ϊlabel
    % segmentNum��������֤���������ֳɶ��ٸ�Ƭ��
    % ���أ�
    % dataSegments�����ֳ�segmentNum��Ƭ�ε�data��segmentNum��totalClass��cell
    % dim��featureά��
    
    totalClass = length(unique(data(: , end))); % ���������
    dim = size(data , 2) - 1; % featureά��
    totalSample = size(data , 1);
    dataSegments = [];
    % labelSegments = [];
    for i = 1 : totalClass % i ���У��ڼ�����
        dataTemp{1 , i} = data(data(: , end) == i , 1 : end - 1);
        % ��ȡdata��label�е���i�������У���ȥ��label��
        ithClassLength = size(dataTemp{1 , i} , 1); % ��i��data������
             
        index = randperm(ithClassLength); % ����i��data˳�������indexΪ[48 , 1 , 50 , 3 ,���� ]
        segSize = floor(ithClassLength/segmentNum); % ��i����ֳ�segSize��segment
        
        % ���´��뽫ÿһ�л��ֳ�segmentNum��
        for j = 1: segmentNum - 1 % �������һ��segment��j���У��ڼ���segment
            dataSegments{j , i} = dataTemp{1 , i}(index(segSize * (j - 1) + 1 : segSize * j) , :);
            % 5 �� 3��cell��3�࣬ÿ��cell����10 �� 4
        end
        dataSegments{j + 1 , i} = dataTemp{1 , i}(index(segSize * j + 1 : ithClassLength) , :);
        % ���һ��segment���ܲ���segSize��������ǵ�ithClassLength
    end
end
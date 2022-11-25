# Precipitation-isotope-prediction-model
(1)K-means++
function [centroid, class] = Kmeanspp(data, k, iteration)
% Main part of Kmeans clustering algorithm.
%
% Args:
%   data: data to be clustered (n * p)
%   k: the number of classes
%   iteration: maximum number of iterations
%
% Returns:
%   centroid: clustering centroids for all classes
%   class: corresponding class for all samples

% Choose the first inital centroid randomly
centroid = data(randperm(size(data,1),1)',:);

% Select remaining initial centroids (a total number of k-1)
for i = 2:k
    distance_matrix = zeros(size(data,1),i-1);
    for j = 1:size(distance_matrix,1)
        for p = 1:size(distance_matrix,2)
            distance_matrix(j,p) = sum((data(j,:)-centroid(p,:)) .^ 2);
        end
    end
    % Choose next centroid according to distances between points and
    % previous cluster centroids.
    index = Roulettemethod(distance_matrix);
    centroid(i,:) = data(index,:);
    clear distance_matrix;
end

% Following steps are same to kmeans
class = zeros(size(data,1),1);
distance_matrix = zeros(size(data,1), k);

for i = 1:iteration
    
    previous_result = class; % for early termination
    
    % Calculate eculidean distance between each sample and each centroid
    for j = 1:size(distance_matrix,1)
        for k = 1:size(distance_matrix,2)
            distance_matrix(j,k) = sqrt((data(j,:)-centroid(k,:)) * (data(j,:)-centroid(k,:))');
        end
    end
    
    % Assign each sample to the nearest controid
    [~,class] = min(distance_matrix,[],2);
    
    % Recalculate centroids
    for j = 1:k
        centroid(j,:) = mean(data(class(:,1) == j,:));
    end
    
    % Display
    fprintf('---- %ith iteration completed---- \n',i);
    
    % If classified results on all points do not change after an iteration, 
    % the clustering process will quit immediately.
    if(class == previous_result)
        fprintf('**** Clustering over after %i iterations ****\n',i);
        break;
    end
end
end

function [index] = Roulettemethod(distance_matrix)

% Find shortest distance between one sample and its closest cluster centroid
[min_distance,~] = min(distance_matrix,[],2);

% Normalize for further operations
min_distance = min_distance ./ sum(min_distance);

% Construct roulette according to min_distance
temp_roulette = zeros(size(distance_matrix,1),1);
for i = 1:size(distance_matrix,1)
    temp_roulette(i,1) = sum(min_distance(1:i,:));
end

% Generate a random number for selection
temp_rand = rand();

% Find the corresponding index
for i = 1:size(temp_roulette,1)
    if((i == 1) && temp_roulette(i,1) > temp_rand)
        index = 1;
    elseif((temp_roulette(i,1) > temp_rand) && (temp_roulette(i-1,1) < temp_rand))
        index = i;
    end
end
end


(2)CNN Model
%% 加载数据与数据集划分
clc;clear;close all
data=xlsread('data');
input =data(:,1:6)';
output=data(:,7)';
nwhole =size(data,1);
ntrain =1114;
%train_ratio=0.9;
%ntrain=round(nwhole*train_ratio);
ntest =nwhole-ntrain;
% 准备输入和输出训练数据
input_train  = input(:,1:ntrain);
output_train = output(:,1:ntrain);
% 准备测试数据
input_test  = input(:, ntrain+1:ntrain+ntest);
output_test = output(:,ntrain+1:ntrain+ntest);


%% 数据归一化
method=@mapminmax;
[inputn_train,inputps]=method(input_train);
inputn_test=method('apply',input_test,inputps);
[outputn_train,outputps]=method(output_train);
outputn_test=method('apply',output_test,outputps);

%% 数据的一个转换，转换成MATLAB的CNN的输入数据形式，是4-D形式的，最后一维就是样本数
trainD=reshape(inputn_train,[size(input,1),1,1,ntrain]);%训练集输入
testD =reshape(inputn_test,  [size(input,1),1,1,ntest]); %测试集输入
targetD       = outputn_train;%训练集输出
targetD_test  = outputn_test;%测试集输出

%% CNN模型建立

layers = [
    imageInputLayer([size(input,1) 1 1])     %输入层参数设置
    convolution2dLayer(3,16,'Padding','same')%卷积层的核大小、数量，填充方式
    reluLayer                                %relu激活函数
    fullyConnectedLayer(1)   % 输出层神经元
    regressionLayer];        % 添加回归层，用于计算损失值
%% 模型训练与测试

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',0.05, ...
    'GradientThreshold',1, ...
    'Verbose',false,...
    'Plots','training-progress');
% 训练
net = trainNetwork(trainD,targetD',layers,options);
%% 训练集误差评价
% 预测
YPred1 = predict(net,trainD); 
YPred1 =double(YPred1');
% 反归一化
CNNoutput_train=method('reverse',YPred1,outputps);
CNNoutput_train=double(CNNoutput_train);
CNNerror_train=CNNoutput_train'-output_train';
CNNpererror_train=CNNerror_train./output_train';
CNNpererror_train=filloutliers(CNNpererror_train,'spline');
CNNpererror_train(abs(CNNpererror_train)>1)=0;
% R
Rtrain21=sum((output_train -mean(output_train)).*( CNNoutput_train - mean(CNNoutput_train)));
Rtrain22=sqrt(sumsqr(output_train -mean(output_train)).*sumsqr(CNNoutput_train - mean(CNNoutput_train)));
Rtrain = Rtrain21/Rtrain22;
% RMSE
RMSEtrain = sqrt(sumsqr(CNNerror_train)/length(output_train));
% MAPE
MAPEtrain = mean(abs(CNNpererror_train));
disp("——————CNN网络模型训练数据——————————")
disp("    预测值     真实值     误差   ")
disp([CNNoutput_train' output_train' CNNerror_train])
%--------------------------------------------------------------------------
%disp('CNN训练平均绝对误差百分比MAPE');
%disp(MAPEtrain)
%disp('CNN训练均方根误差RMSE')
%disp(RMSEtrain)
%--------------------------------------------------------------------------
% 数据可视化
figure()
plot(CNNoutput_train,':.')  
hold on
plot(output_train,'-')           
legend( '训练数据','实际数据','Location','NorthWest','FontName','华文宋体');
title('CNN模型训练结果及真实值','fontsize',12,'FontName','华文宋体')
xlabel('时间','fontsize',12,'FontName','华文宋体');
ylabel('数值','fontsize',12,'FontName','华文宋体');
xlim([1 ntrain]);
%-------------------------------------------------------------------------------------
figure()
plot(CNNerror_train,'-','Color',[128 0 0]./255,'linewidth',1)   
legend('CNN模型训练误差','Location','NorthEast','FontName','华文宋体')
title('CNN模型训练误差','fontsize',12,'FontName','华文宋体')
ylabel('误差','fontsize',12,'FontName','华文宋体')
xlabel('样本','fontsize',12,'FontName','华文宋体')
xlim([1 ntrain]);
%-------------------------------------------------------------------------------------
figure()
plot(CNNpererror_train,'-','Color',[128 0 255]./255,'linewidth',1)   
legend('CNN模型训练相对误差','Location','NorthEast','FontName','华文宋体')
title('CNN模型训练相对误差','fontsize',12,'FontName','华文宋体')
ylabel('误差','fontsize',12,'FontName','华文宋体')
xlabel('样本','fontsize',12,'FontName','华文宋体')
xlim([1 ntrain]);
%% 测试集误差评价
% 预测
YPred = predict(net,testD); 
% 结果
YPred =double(YPred');
% 反归一化
CNNoutput_test=method('reverse',YPred,outputps);
CNNoutput_test=double(CNNoutput_test);

CNNerror_test=CNNoutput_test'-output_test';
CNNpererror_test=CNNerror_test./output_test';
CNNpererror_test=filloutliers(CNNpererror_test,'spline');
CNNpererror_test(abs(CNNpererror_test)>1)=0;
% R
Rtest21=sum((output_test -mean(output_test)).*( CNNoutput_test - mean(CNNoutput_test)));
Rtest22=sqrt(sumsqr(output_test -mean(output_test)).*sumsqr(CNNoutput_test - mean(CNNoutput_test)));
Rtest = Rtest21/Rtest22;
% RMSE
RMSEtest = sqrt(sumsqr(CNNerror_test)/length(output_test));
% MAE
MAEtest = mean(abs(CNNerror_test));
% MAPE
MAPEtest = mean(abs(CNNpererror_test));
% R^2
test=(Rtest)^2;

disp("——————CNN网络模型测试数据——————————")
disp("    预测值     真实值     误差   ")
disp([CNNoutput_test' output_test' CNNerror_test])
%--------------------------------------------------------------------------

disp('CNN测试均方根误差RMSE')
disp(RMSEtest)
disp('CNN网络预测绝对平均误差MAE');
disp(MAEtest);
disp('CNN决定系数R^2')
disp(test)
disp('CNN相关系数R')
disp(Rtest);
disp('CNN测试平均绝对误差百分比MAPE');
disp(MAPEtest);
%--------------------------------------------------------------------------

(3)Bi-LSTM Model
%%
clc;clear;
warning off;
%% 导入数据
data=xlsread('data');
% 输入数据
input =data(:,1:6)';
output=data(:,7)';
nwhole =size(data,1);
trainNum =1114;
testNum = nwhole - trainNum;  
%train_ratio=0.9;
%ntrain=round(nwhole*train_ratio);
%ntest =nwhole-ntrain;
%temp=randperm(nwhole);
% 准备输入和输出训练数据
input_train =input(:, 1:trainNum);
output_train=output(:,1:trainNum);
% 准备测试数据
input_test =input(:, trainNum+1:trainNum+testNum);
output_test=output(:,trainNum+1:trainNum+testNum);

%% 归一化（全部特征 均归一化）
[inputn_train,inputps]  =mapminmax(input_train);
[outputn_train,outputps]=mapminmax(output_train);
inputn_test =mapminmax('apply',input_test,inputps); 
outputn_test=mapminmax('apply',output_test,outputps); 
%% BiLSTM 层设置，参数设置
inputSize  = size(inputn_train,1);   %数据输入x的特征维度
outputSize = size(outputn_train,1);  %数据输出y的维度  
numhidden_units=180;
%% BiLSTM
layers = [ ...
    sequenceInputLayer(inputSize)                 %输入层设置
    fullyConnectedLayer(inputSize) 
    bilstmLayer(numhidden_units)                  %学习层设置(cell层）
    dropoutLayer(0.2)
    bilstmLayer(numhidden_units)
    dropoutLayer(0.2)
    fullyConnectedLayer(outputSize)               % 全连接层设置（影响输出维度）
    regressionLayer('name','out')];
%% trainoption(bilstm)
opts = trainingOptions('adam', ...
    'MaxEpochs',60, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...                % epoch后学习率更新
    'LearnRateDropFactor',0.8, ...
    'Verbose',0, ...
    'Plots','training-progress'... 
    );

%% BiLSTM网络训练
tic
BiLSTMnet = trainNetwork(inputn_train ,outputn_train ,layers,opts);
toc;
[BiLSTMnet,BiLSTMoutputr_train]= predictAndUpdateState(BiLSTMnet,inputn_train);
BiLSTMoutput_train = mapminmax('reverse',BiLSTMoutputr_train,outputps);
%% BiLSTM测试数据
%%
%网络测试输出
[BiLSTMnet,BiLSTMoutputr_test] = predictAndUpdateState(BiLSTMnet,inputn_test);
%网络输出反归一化
BiLSTMoutput_test= mapminmax('reverse',BiLSTMoutputr_test,outputps);
%% BiLSTM数据输出
%%
%-------------------------------------------------------------------------------------
error_train=BiLSTMoutput_train'-output_train';
pererror_train=error_train./output_train';
error_test=BiLSTMoutput_test'-output_test';
pererror_test=error_test./output_test';
%pererror=pererror_test';
pererror=pererror_test*100';
error=error_test';
avererror=sum(abs(error))/(testNum);
averpererror=sum(abs(pererror))/(testNum);
% RMSE
RMSEtest = sqrt(sumsqr(error_test)/length(output_test));
% MAE
MAEtest = mean(abs(error_test));
% MAPE
MAPEtest = mean(abs(pererror_test));
% R
Rtest21=sum((output_test -mean(output_test)).*( BiLSTMoutput_test - mean(BiLSTMoutput_test)));
Rtest22=sqrt(sumsqr(output_test -mean(output_test)).*sumsqr(BiLSTMoutput_test - mean(BiLSTMoutput_test)));
Rtest = Rtest21/Rtest22;
% R^2
test=(Rtest)^2;

disp('BiLSTM测试均方根误差RMSE')
disp(RMSEtest)
disp('BiLSTM网络预测绝对平均误差MAE');
disp(MAEtest);
disp('BiLSTM决定系数R^2')
disp(test)
disp('BiLSTM相关系数R')
disp(Rtest);
disp('BiLSTM测试平均绝对误差百分比MAPE');
disp(MAPEtest);

(4)CNN-Bi-LSTM Model
%% CNN-BiLSTM多变量回归预测
%% 加载数据与数据集划分
clc;clear;close all
data = xlsread('data');
% 输入数据
input =data(:,1:6)';
output=data(:,7)';
nwhole =size(data,1);
% 打乱数据集
% temp=randperm(nwhole);
% 不打乱数据集
temp=1:nwhole;
ntrain =628;
%train_ratio=0.5;
%ntrain=round(nwhole*train_ratio);
ntest =nwhole-ntrain;
% 准备输入和输出训练数据
input_train =input(:,temp(1:ntrain));
output_train=output(:,temp(1:ntrain));
% 准备测试数据
input_test =input(:, temp(ntrain+1:ntrain+ntest));
output_test=output(:,temp(ntrain+1:ntrain+ntest));
%% 数据归一化
method=@mapminmax;
[inputn_train,inputps]=method(input_train);
inputn_test=method('apply',input_test,inputps);
[outputn_train,outputps]=method(output_train);
outputn_test=method('apply',output_test,outputps);
% 创建元胞或向量，长度为训练集大小；
XrTrain = cell(size(inputn_train,2),1);
YrTrain = zeros(size(outputn_train,2),1);
for i=1:size(inputn_train,2)
    XrTrain{i,1} = inputn_train(:,i);
    YrTrain(i,1) = outputn_train(:,i);
end
% 创建元胞或向量，长度为测试集大小；
XrTest = cell(size(inputn_test,2),1);
YrTest = zeros(size(outputn_test,2),1);
for i=1:size(input_test,2)
    XrTest{i,1} = inputn_test(:,i);
    YrTest(i,1) = outputn_test(:,i);
end

%% 创建混合CNN-BiLSTM网络架构
% 输入特征维度
numFeatures  = size(inputn_train,1);
% 输出特征维度
numResponses = 1;
FiltZise = 10;
%  创建"CNN-BiLSTM"模型
    layers = [...
        % 输入特征
        sequenceInputLayer([numFeatures 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')
        % CNN特征提取
        convolution2dLayer([FiltZise 1],32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
        % 展开层
        sequenceUnfoldingLayer('Name','unfold')
        % 平滑层
        flattenLayer('Name','flatten')
        % BiLSTM特征学习
        bilstmLayer(128,'Name','bilstm1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        % BiLSTM输出
        bilstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        % 全连接层
        fullyConnectedLayer(numResponses,'Name','fc')
        regressionLayer('Name','output')    ];

    layers = layerGraph(layers);
    layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

%% CNNBiLSTM训练选项
% 批处理样本
MiniBatchSize =24;
% 最大迭代次数
MaxEpochs = 500;
% 学习率
learningrate = 0.005;
% 一些参数调整
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end
    options = trainingOptions( 'adam', ...
        'MaxEpochs',60, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',20, ...
        'LearnRateDropFactor',0.8, ...
        'L2Regularization',1e-3,...
        'Verbose',false, ...
        'ExecutionEnvironment',mydevice,...
        'Plots','training-progress');

%% 训练混合网络
% rng(0);
% 训练
net = trainNetwork(XrTrain,YrTrain,layers,options);
% 预测
YPred = predict(net,XrTest,"ExecutionEnvironment",mydevice,"MiniBatchSize",numFeatures);
% 结果
YPred =double(YPred');
% 反归一化
CNNBiLSTMoutput_test=method('reverse',YPred,outputps);
CNNBiLSTMoutput_test=double(CNNBiLSTMoutput_test);
%% 测试集误差评价
CNNBiLSTMerror_test=CNNBiLSTMoutput_test'-output_test';
CNNBiLSTMpererror_test=CNNBiLSTMerror_test./output_test';
% RMSE
RMSEtest = sqrt(sumsqr(CNNBiLSTMerror_test)/length(output_test));
% MAE
MAEtest = mean(abs(CNNBiLSTMerror_test));
% MAPE
MAPEtest = mean(abs(CNNBiLSTMpererror_test));
r = corrcoef(output_test,CNNBiLSTMoutput_test);
% R
Rtest21=sum((output_test -mean(output_test)).*( CNNBiLSTMoutput_test - mean(CNNBiLSTMoutput_test)));
Rtest22=sqrt(sumsqr(output_test -mean(output_test)).*sumsqr(CNNBiLSTMoutput_test - mean(CNNBiLSTMoutput_test)));
Rtest = Rtest21/Rtest22;
% R^2
test=(Rtest)^2;
disp("——————CNNBiLSTM网络模型测试数据——————————")
disp("    预测值     真实值     误差   ")
disp([CNNBiLSTMoutput_test' output_test' CNNBiLSTMerror_test])
%--------------------------------------------------------------------------
disp('CNNBiLSTM测试均方根误差RMSE')
disp(RMSEtest)
disp('CNNBiLSTM测试平均绝对误差MAE');
disp(MAEtest)
disp('CNNBiLSTM决定系数R^2')
disp(test)
disp('CNNBiLSTM相关系数R')
disp(Rtest);
disp('CNNBiLSTM测试平均绝对误差百分比MAPE');
disp(MAPEtest)%--------------------------------------------------------------------------

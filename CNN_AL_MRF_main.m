
%% Experiment for HSIs classification using CNNs, active learning and MRF incorporated model
% writen by ...
% Aug. 2019.
% the folder containing this function is suggested to be located under ...\matconvnet\examples\
%
% if you find this code useful, please kindly cite the following papers:
%       [1]
%       [2]
%       ...

close all; clear, clc;
Dir.Cur = pwd;
addpath(genpath(Dir.Cur));

%% Parameters for data
data.NameFolder = {'IndianPines', 'PaviaU', 'PaviaCenter'};
data.NameMat = {'GT.mat', 'Feature.mat'};
data.SizeOri = {[145, 145, 220], [610, 340, 103], [400, 300, 102]};
data.SizeWin = 8;
data.NumClass = {16, 9, 8};
data.IndBand = {[10, 80, 200], [12, 67, 98], [10, 60, 90]}; % to generate false RGB, which should be less contaminated bands
%data.flagPCA = true;
%data.ReducedDim = 10;

% Three datasets:
% data.flagSet = 1, Indian Pines; 
%              = 2, Pavia University;
%              = 3, Pavia Center.
data.flagSet = 1;

data.NameFolder = data.NameFolder{data.flagSet};
data.SizeOri = data.SizeOri{data.flagSet};
data.NumClass = data.NumClass{data.flagSet};
data.IndBand = data.IndBand{data.flagSet};

%% Parameters for algorithm
alg.SampleSty = 'Rd'; % out of {'Rd', 'Classwise'}
alg.CountSty = 'Num'; % out of {'Num', 'Ratio'}
alg.NumTrn1st = {250, 107, 58};
alg.NumTrn1st = alg.NumTrn1st{data.flagSet};
% if alg.CountSty == 'Ratio'
%alg.RatioTrn1st = {0.02, 0.0025, 0.0025};
%alg.RatioTrn1st = alg.RatioTrn1st{data.flagSet};
alg.CrossVal = 0.05;
alg.NumAlAugPerIte = {[250, 150, 100, 50], [107, 107, 107], [26, 20]}; % The training samples added in each iteration keeps the same ratio with the training sample number of the first iteration
alg.NumAlAugPerIte = alg.NumAlAugPerIte{data.flagSet};
alg.NumIter = length(alg.NumAlAugPerIte)+1;
alg.AlStra = 'BvSB'; % out of {'BvSB', 'RS', 'EP'};
alg.SmoothFactor = 8; % around 10
alg.flagAL = true;
alg.flagMRF = true;

alg.RngSeed = 4;

%% Parameters for CNN module
cnn_net.NumEpoch = {[800, 400, 400, 300, 200], [400, 200, 200, 200], [600, 400, 300]}; % Finetune helps to coverge faster
cnn_net.SizeKer1Conv = 3;       cnn_net.SizeKer2Conv = 2;
cnn_net.NumKer1Conv  = 20;      cnn_net.NumKer2Conv  = 20; 
cnn_net.SizeKer1Pool = 2;       cnn_net.SizeKer2Pool = 2;
cnn_net.Stride1Pool  = 2;       cnn_net.Stride2Pool  = 2;
cnn_net.SizeKer1FC   = 2;       cnn_net.SizeKer2FC   = 1;
cnn_net.NumKer1FC    = 500;     cnn_net.NumKer2FC    = data.NumClass;
cnn_net.WeightDecay  = 0.0005;  cnn_net.BatchSize    = 50;
cnn_net.LearningRate = 0.001; % or logspace(-3, -4, cnn_net.NumEpoch(Ite))
cnn_net.flagDA = true;          cnn_net.flagBN = true;
cnn_net.flagFT = true;

if cnn_net.flagFT
    cnn_net.NumEpoch = cnn_net.NumEpoch{data.flagSet};
else
    cnn_net.NumEpoch = [800, 400, 600];
    cnn_net.NumEpoch = cnn_net.NumEpoch(data.flagSet);
end

data = rmfield(data, 'flagSet');

%% make folders
cd ..
cd ..
Dir.Root = pwd;
Dir.Exp = strcat(pwd, '\data\HSI\', data.NameFolder, '\RngSeed-', num2str(alg.RngSeed));
if ~cnn_net.flagDA
    Dir.Exp = fullfile(Dir.Exp, 'CNN');
else if ~cnn_net.flagBN
        Dir.Exp = fullfile(Dir.Exp, 'DA');
    else if ~alg.flagAL
            Dir.Exp = fullfile(Dir.Exp, 'BN');
        else if ~cnn_net.flagFT
                Dir.Exp = fullfile(Dir.Exp, 'AL');
            else if alg.flagMRF
                Dir.Exp = fullfile(Dir.Exp, 'FT-MRF');
                end
            end
        end
    end
end
Dir.Results = fullfile(Dir.Exp, 'Results');

if ~exist(Dir.Exp) || ~exist(Dir.Results)
    mkdir(Dir.Exp);
    mkdir(Dir.Results);
end

cd(Dir.Cur)
Dir.Data = fullfile(pwd, 'Data', data.NameFolder);

%% Generate initialized labeled pixels
[data, Train, Patch] = CNN_AL_MRF_preprocess(Dir, data, alg);

% initialize
Train.Set_All = [];
Train.Data_Train = [];  Train.Set_Train = [];
Train.Data_Val = [];    Train.Set_Val = [];

%% Iterations of the overall algorithm, including data preparation, CNNs' training and testing (with active learning), post-preprossing with or w/o. MRF
for Ite = 1:alg.NumIter

    %% step 1: data preparation (with data augmentation)
    [Train, Test] = CNN_AL_MRF_prepdata(data, cnn_net, alg, Train, Patch);

    %% step 2: train CNNs
    [net] = CNN_AL_MRF_train(Dir, data, cnn_net, Train, Ite, Test);

    %% step 3: test
    [Train] = CNN_AL_MRF_test(Dir, data, alg, cnn_net, Train, Ite, Test);
end
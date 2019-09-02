function [Train, Test] = CNN_AL_MRF_prepdata(data, cnn_net, alg, Train, Patch)

% update train set for all
Train.Set_All = [Train.Set_All, Train.Set];

% update train pool
for i = 1:size(Train.Set, 2)
    Train.Pool(:, find(Train.Pool(1, :) == Train.Set(1, i))) = [];
end

% Construct Test Dataset: load padded feature into #Test_num voxels
Test.Set = Train.Pool;
Test.Num = size(Train.Pool, 2);
Test.Data = single(zeros(data.SizeWin, data.SizeWin, size(data.F_pad,3), Test.Num));

for i=1:Test.Num
    [x, y] = Index1Dto2D(Test.Set(1,i), data.SizeOri(1), data.SizeOri(2));
    % patch center index: x+HalfWin, y+HalfWin
    Test.Data(:,:,:,i) = data.F_pad(x+Patch.Start:x+Patch.End, y+Patch.Start:y+Patch.End, :);
end

% construct Train Dataset: load padded feature into #Train_num voxels
Train_num = size(Train.Set, 2);
Train_Data = single(zeros(data.SizeWin, data.SizeWin, size(data.F_pad,3), Train_num));

for i=1:Train_num
    [x, y] = Index1Dto2D(Train.Set(1,i), data.SizeOri(1), data.SizeOri(2));
    Train_Data(:,:,:,i) = data.F_pad(x+Patch.Start:x+Patch.End, y+Patch.Start:y+Patch.End, :);
end

% index partition for train and val.(in train dataset)
Val_num = floor(alg.CrossVal * Train_num);
tmp = randperm(Train_num);
Val_Idx = tmp(1:Val_num);
Train_Idx = tmp(Val_num+1: end);

% construct original train and cross validation data
Train_Data_Val = Train_Data(:, :, :, Val_Idx);
Train_Set_Val = Train.Set(:, Val_Idx);

Train_Data_Train = Train_Data(: ,:, :, Train_Idx);
Train_Set_Train = Train.Set(:, Train_Idx);

if cnn_net.flagDA
% Augment the data by rotating and flipping
    Val_Data_1 = rot90(Train_Data_Val, 1);
    Val_Data_2 = rot90(Train_Data_Val, 2);
    Val_Data_3 = rot90(Train_Data_Val, 3);
    Val_Data_4 = flipud(Train_Data_Val);
    Val_Data_5 = fliplr(Train_Data_Val);
    
    Train_Data_Val = cat(4, Train_Data_Val, Val_Data_1, Val_Data_2, Val_Data_3, Val_Data_4, Val_Data_5);
    Train_Set_Val = repmat(Train_Set_Val, 1, 6);

    Train_Data_1 = rot90(Train_Data_Train, 1);
    Train_Data_2 = rot90(Train_Data_Train, 2);
    Train_Data_3 = rot90(Train_Data_Train, 3);
    Train_Data_4 = flipud(Train_Data_Train);
    Train_Data_5 = fliplr(Train_Data_Train);
    
    Train_Data_Train = cat(4, Train_Data_Train, Train_Data_1, Train_Data_2, Train_Data_3, Train_Data_4, Train_Data_5);
    Train_Set_Train = repmat(Train_Set_Train, 1, 6);
end

% update Train
Train.Data_Val = cat(4, Train.Data_Val, Train_Data_Val);
Train.Set_Val = cat(2, Train.Set_Val, Train_Set_Val);
Train.Data_Train = cat(4, Train.Data_Train, Train_Data_Train);
Train.Set_Train = cat(2, Train.Set_Train, Train_Set_Train);

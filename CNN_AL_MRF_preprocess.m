function [data, Train, Patch] = CNN_AL_MRF_preprocess(Dir, data, alg)

Img_Size = data.SizeOri(1:2);

% Load feature and ground truth labels: Feature.mat, GT.mat
load(fullfile(Dir.Data, data.NameMat{1})); 
load(fullfile(Dir.Data, data.NameMat{2}));
GT = double(GT);
data.GT = GT;

% if data.flagPCA
%     data.F = Feature;
%     % PCA Dim Reduction
%     Feature = reshape(Feature, [prod(lw), data.SizeOri(3)]);
%     [eigvec, ~] = PCA(Feature, data);
%     Feature = Feature*eigvec; 
%     Feature = reshape(Feature, [lw, data.ReducedDim]);
%     % normalize
%     for b = 1:data.ReducedDim
%         Feature(:, :, b) = (Feature(:, :, b) - min(min(Feature(:, :, b))))./(max(max(Feature(:, :, b))) ...
%             - min(min(Feature(:, :, b))));
%     end
% else
%     for b = 1:data.SizeOri(3)
%         Feature(:, :, b) = (Feature(:, :, b) - min(min(Feature(:, :, b))))./(max(max(Feature(:, :, b))) ...
%             - min(min(Feature(:, :, b))));
%     end
%     data.F_norm = Feature;
% end

for b = 1:data.SizeOri(3)
    Feature(:, :, b) = (Feature(:, :, b) - min(min(Feature(:, :, b))))./(max(max(Feature(:, :, b))) ...
        - min(min(Feature(:, :, b))));
end
data.F_norm = Feature;

% padding
% if data.SizeWin is even£¬left padding HalfWin£¬right padding HalfWin-1
% if data.SizeWin is odd£¬left padding HalfWin£¬right padding HalfWin
HalfWin = floor(data.SizeWin/2);
if (mod(data.SizeWin, 2) == 0)
    Patch.Start = HalfWin-HalfWin;
    Patch.End = HalfWin+HalfWin-1;
else
    Patch.Start = HalfWin-HalfWin;
    Patch.End = HalfWin+HalfWin;
end
data.F_pad = single(zeros(Img_Size(1)+Patch.End, Img_Size(2)+Patch.End, size(Feature, 3)));
data.F_pad(HalfWin+1:Img_Size(1)+HalfWin, HalfWin+1:Img_Size(2)+HalfWin, :) = Feature;

data.MRF = single(zeros(data.SizeWin, data.SizeWin, size(data.F_pad,3), prod(Img_Size)));
if alg.flagMRF
    for i=1:prod(Img_Size)
        [x, y] = Index1Dto2D(i, Img_Size(1), Img_Size(2));
        data.MRF(:,:,:,i) = data.F_pad(x+Patch.Start:x+Patch.End, y+Patch.Start:y+Patch.End, :);
    end
end

% randomly select the training set
GT_1d = data.GT(:)';
GT_indexes = find(GT_1d);
Train.Pool = [GT_indexes; GT_1d(GT_indexes)];

% randomly sampling with seed
rng(alg.RngSeed), tmp = randperm(length(GT_indexes));
tmp_indexes = GT_indexes(tmp);
tmp_GT = GT_1d(tmp_indexes);
if strcmp(alg.SampleSty, 'Classwise') && strcmp(alg.CountSty, 'Ratio')
    Train.Set = [];
    for l=1:max(tmp_GT)
        Train_GlbIndexes = tmp_indexes(find(tmp_GT==l));
        %tmp = max(floor(length(Train_GlbIndexes)*alg.RatioTrn1st), 2);
        tmp = round(length(Train_GlbIndexes)*alg.RatioTrn1st);
        Train_Set = [Train_GlbIndexes(1:tmp);GT_1d(Train_GlbIndexes(1:tmp))];
        Train.Set = [Train.Set, Train_Set];
    end
elseif strcmp(alg.SampleSty, 'Rd') && strcmp(alg.CountSty, 'Num')
    Train_GlbIndexes = tmp_indexes(1:alg.NumTrn1st);
    Train.Set = [Train_GlbIndexes; GT_1d(Train_GlbIndexes)];
end
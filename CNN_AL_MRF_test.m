function [Train] = CNN_AL_MRF_test(Dir, data, alg, cnn_net, Train, Ite, Test)

Img_Size = data.SizeOri(1:2);
% load test data
load(strcat(Dir.Exp, '\imdb.mat'));

test_index = find(images.set==3);
test_data = images.data(:,:,:,test_index);

% load current network
if ~cnn_net.flagFT
    epks = num2str(cnn_net.NumEpoch(Ite));
    load(strcat(Dir.Exp, '\net-epoch-',epks,'.mat'));
    epks = num2str(sum(cnn_net.NumEpoch(1:Ite)));
else 
    epks = num2str(sum(cnn_net.NumEpoch(1:Ite)));
    load(strcat(Dir.Exp, '\net-epoch-',epks,'.mat'));
end
% change 'softmaxloss' with 'softmax'
net.layers{1, end}.type = 'softmax';

% test
Result = zeros(1, prod(Img_Size));
Data_Cost = zeros(prod(Img_Size), data.NumClass);
for i = 1:length(test_index)
    im_ = test_data(:, :, :, i);
    res = vl_simplenn(net, im_, [], [], 'mode', 'test');
    scores = squeeze(gather(res(end).x));
    [~,  best] = max(scores);
    Data_Cost(Test.Set(1,i), :) = scores + 1e-38;
    Result(Test.Set(1,i)) = best;
end

% Except for the final epoch, add new samples to train set
if (Ite < alg.NumIter)
    if strcmp(alg.AlStra, 'BvSB') % by active learning
        ALIncr = Data_Cost(Train.Pool(1,:), :);
        ALIncrsort = sort(ALIncr, 2, 'descend');                          
        ALIncrsort_MinBT = ALIncrsort(:, 1) - ALIncrsort(:, 2);
        [~, indexsortminppBT] = sort(ALIncrsort_MinBT);
        xp = indexsortminppBT(1:alg.NumAlAugPerIte(Ite));
    elseif strcmp(alg.AlStra, 'RS')% by random sampling
        rng(alg.RngSeed), tmp = randperm(size(Train.Pool, 2));
        xp = tmp(1:alg.NumAlAugPerIte(Ite)); % ###
    elseif strcmp(alg.AlStra, 'EP')
        ALIncr = Data_Cost(Train.Pool(1,:), :);
        for m = 1:size(ALIncr,1)
            ALEntropy(m) = -sum(ALIncr(m,:).*log(ALIncr(m,:)));
        end
        [~, indexsortALEntropy] = sort(ALEntropy, 'descend');
        xp = indexsortALEntropy(1:alg.NumAlAugPerIte(Ite));
    end
    Train.Set = Train.Pool(:,xp);
else
    if alg.flagMRF
        for i = 1:prod(Img_Size)
            im_ = data.MRF(:, :, :, i)-images.data_mean;
            res = vl_simplenn(net, im_, [], [], 'mode', 'test');
            scores = squeeze(gather(res(end).x));
            Data_Cost(i, :) = scores + 1e-38;
        end
        Data_Cost = -log(Data_Cost);
        DataCost_Path = strcat(Dir.Exp, '\DataCost-epoch', epks, '-DataCost.txt');
        save(DataCost_Path, '-ascii', 'Data_Cost');

        WeitHor_Path = strcat(Dir.Data, '\HorzWeight.txt');
        WeitVer_Path = strcat(Dir.Data, '\VertWeight.txt');
        if (~exist(WeitHor_Path, 'file') && ~exist(WeitVer_Path, 'file'))
            CNN_AL_MRF_GenWeit(data, WeitHor_Path, WeitVer_Path);
        end
        DataCost_Path = strcat(Dir.Exp, '\DataCost-epoch', epks, '-DataCost.txt');
        ResultMRF_Path = strcat(Dir.Exp, '\Epoch', epks, '-ResultMRF', num2str(alg.SmoothFactor), '.txt');    
        % post-precessing with MRF
        system([strcat(Dir.Cur, '\utils\mrf.exe'), ' ',num2str(Img_Size(1)), ' ', num2str(Img_Size(2)), ' ', ...
            num2str(data.NumClass), ' ', DataCost_Path, ' ', WeitHor_Path, ' ', WeitVer_Path, ' ', ...
            ResultMRF_Path, ' ', num2str(alg.SmoothFactor)]);
        % other necessary processings
        Result_MRF = load(ResultMRF_Path);
        Result_MRF = Result_MRF';
        Result_MRF = Result_MRF(:).*logical(data.GT(:));
        
        %Result = Result_MRF(Test.Set(1,:))';
    else
        Result = Result(find(Result));
    end
    acc = length(find(Result==Test.Set(2,:)))/Test.Num;
    disp(['Epoch = ', epks, ', OA = ', num2str(acc*100), '%']);
    save(strcat(Dir.Results, '\Result-OA', epks, '-Ite', num2str(Ite), '.mat'), 'Result');
    fid = fopen(strcat(Dir.Results, '\Accuracy.txt'), 'a+');
    fprintf(fid, 'Ite:%d, Epoch:%d, \nOA: %-8.4f%%\r\n', Ite, str2num(epks), acc*100);    
    
    acc = [];
    for l=1:max(Result)
        a = find(Result==l);
        b = Test.Set(1, find(Test.Set(2,:)==l)); 
        acc(l) = length(intersect(a,b))/length(b);
        fprintf(fid, 'Class #%d: %-8.4f%%\r\n', l, acc(l)*100);
    end
    disp(['Epoch = ', epks, ', AA = ', num2str(mean(acc)*100), '%']);
    fclose(fid);
end
end

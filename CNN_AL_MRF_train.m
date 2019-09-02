function [net] = CNN_AL_MRF_train(Dir, data, cnn_net, Train, Ite, Test)
                                    
% based on MatConvNet
run(fullfile(Dir.Root, '\matlab\vl_setupnn.m'));

if ~isfield(cnn_net, 'gpus'), cnn_net.gpus = []; end;

if Ite==1 || ~cnn_net.flagFT % wait to be simplified
    [net] = CNN_AL_MRF_init(data, cnn_net, Train);
else
    % load current network
    epks = num2str(sum(cnn_net.NumEpoch(1:Ite-1)));
    load(strcat(Dir.Exp, '\net-epoch-',epks,'.mat'));
end

if cnn_net.flagFT
    net.meta.trainOpts.numEpochs = sum(cnn_net.NumEpoch(1:Ite));
else
    net.meta.trainOpts.numEpochs = cnn_net.NumEpoch(Ite);
end

imdb = getHSIImdb(data, Train, Test);   % get and save imdb
save(fullfile(Dir.Exp, 'imdb.mat'), '-struct', 'imdb', '-v7.3') ; % renew in each iteration

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x), 1:data.NumClass, 'UniformOutput', false);

trainfn = @cnn_train;
fn = @(x,y) getSimpleNNBatch(x,y);

[net, ~] = trainfn(net, imdb, fn, ...
    'expDir', Dir.Exp, ...
    net.meta.trainOpts, ...
    'gpus', cnn_net.gpus, ...
    'continue', cnn_net.flagFT, ...
    'val', find(imdb.images.set == 2)); % 2: for val; 3: for test

% resume, if needed
% [net, ~] = trainfn(net, imdb, fn, ...
%     'expDir', Dir.Exp, ...
%     net.meta.trainOpts, ...
%     'gpus', cnn_net.gpus, ...
%     'continue', true, ...
%     'val', find(imdb.images.set == 2));

function [images, labels] = getSimpleNNBatch(imdb, batch)

images = imdb.images.data(:,:,:,batch); % return train data
labels = imdb.images.labels(1, batch); % indexs

function imdb = getHSIImdb(data, Train, Test)

% set = 1, train; 
%       2, val;
%       3, test.
set = [ones(1,size(Train.Set_Train, 2)), 2*ones(1,size(Train.Set_Val, 2)), 3*ones(1,size(Test.Set, 2))];

% save data into imdb
imdb.images.data = single(cat(4, Train.Data_Train, Train.Data_Val, Test.Data));
imdb.images.data_mean = mean(imdb.images.data(:, :, :, set==1), 4);
imdb.images.data = bsxfun(@minus, imdb.images.data, imdb.images.data_mean);
imdb.images.labels = single(cat(2, Train.Set_Train(2,:), Train.Set_Val(2,:), Test.Set(2, :)));
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'}; 
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x), 1:data.NumClass, 'uniformoutput', false) ;

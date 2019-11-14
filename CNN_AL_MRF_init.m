function [net] = CNN_AL_MRF_init(data, cnn_net, Train)

rng('default');
rng(0);

% initialize network structure with LeNet5
f = 1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...       % C1: 4-d weights with zero bias
                           'weights', {{f*randn(cnn_net.SizeKer1Conv, cnn_net.SizeKer1Conv, size(Train.Data_Train,3), cnn_net.NumKer1Conv, 'single'), zeros(1, cnn_net.NumKer1Conv, 'single')}}, ... 
                           'stride', 1, ...
                           'pad', 2);
net.layers{end+1} = struct('type', 'pool', ...       % P1 
                           'method', 'max', ...
                           'pool', [cnn_net.SizeKer1Pool, cnn_net.SizeKer1Pool], ...        %池化核大小为2*2
                           'stride', cnn_net.Stride1Pool, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'conv', ...       % C2
                           'weights', {{f*randn(cnn_net.SizeKer2Conv, cnn_net.SizeKer2Conv, cnn_net.NumKer1Conv, cnn_net.NumKer2Conv, 'single'), zeros(1,cnn_net.NumKer2Conv, 'single')}}, ... 
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'pool', ...       % P2 
                           'method', 'max', ...
                           'pool', [cnn_net.SizeKer2Pool, cnn_net.SizeKer2Pool], ...        %池化核大小为2*2
                           'stride', cnn_net.Stride2Pool, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'conv', ...       % FC1
                           'weights', {{f*randn(cnn_net.SizeKer1FC, cnn_net.SizeKer1FC, cnn_net.NumKer2Conv, cnn_net.NumKer1FC, 'single'), zeros(1,cnn_net.NumKer1FC,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'relu') ;         % ReLu 
net.layers{end+1} = struct('type', 'conv', ...       % FC1
                           'weights', {{f*randn(cnn_net.SizeKer2FC, cnn_net.SizeKer2FC, cnn_net.NumKer1FC, cnn_net.NumKer2FC, 'single'), zeros(1,cnn_net.NumKer2FC,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'softmaxloss');  % softmax

% optionally switch to batch normalization
if cnn_net.flagBN
  net = insertBnorm(net, 1);
  net = insertBnorm(net, 4);
  net = insertBnorm(net, 7);
end

% Meta parameters
net.meta.inputSize = [data.SizeWin, data.SizeWin, size(Train.Data_Train,3)];
net.meta.trainOpts.learningRate = cnn_net.LearningRate;
net.meta.trainOpts.weightDecay = cnn_net.WeightDecay;
net.meta.trainOpts.batchSize = cnn_net.BatchSize;

% Fill in defaul values
net = vl_simplenn_tidy(net);
end

function net = insertBnorm(net, l)
% --------------------------------------------------------------  
% insert Bnorm layer between layer l and layer l+1
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));                             % assert layer l has weights
ndim = size(net.layers{l}.weights{1}, 4);                              % neuron number in layer l  
layer = struct('type', 'bnorm', ...                                    % initialize Bnorm parameters
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...                         % weights of Bnorm layer = #neuron of layer l
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;    % add Bnorm layer
end
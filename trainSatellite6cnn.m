clc; clear all; close all;

load Satelliteimages4D.mat;
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
% 
% helperCIFAR10Data.download(url, cifar10Data);

% Load the CIFAR-10 training and test data. 
% [trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);

%%
% Each image is a 32x32 RGB image and there are 50,000 training samples.
size(trainingImages)

%%
% CIFAR-10 has 10 image categories. List the image categories:
numImageCategories = 6;
categories(trainingLabels)

%%

% Display a few of the training images, resizing them for display.
figure
thumbnails = trainingImages(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails)

% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)
% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
% The first convolutional layer has a bank of 32 5x5x3 filters. A
% symmetric padding of 2 pixels is added to ensure that image borders
% are included in the processing. This is important to avoid
% information at the borders being washed away too early in the
% network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

% Note that the third dimension of the filter can be omitted because it
% is automatically deduced based on the connectivity of the network. In
% this case because this layer follows the image layer, the third
% dimension must be 3 to match the number of channels in the input
% image.

% Next add the ReLU layer:
reluLayer()

% Follow it with a max pooling layer that has a 3x3 spatial pooling area
% and a stride of 2 pixels. This down-samples the data dimensions from
% 32x32 to 15x15.
maxPooling2dLayer(3, 'Stride', 2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

]

%%
% A deeper network may be created by repeating these 3 basic layers.
% However, the number of pooling layers should be reduced to avoid
% downsampling the data prematurely. Downsampling early in the network
% discards image information that is useful for learning.
% 
% The final layers of a CNN are typically composed of fully connected
% layers and a softmax loss layer. 

finalLayers = [
    
% Add a fully connected layer with 64 output neurons. The output size of
% this layer will be an array with a length of 64.
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce 10 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
]

%%
% Combine the input, middle, and final layers.
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%%
% Initialize the first convolutional layer weights using normally
% distributed random numbers with standard deviation of 0.0001. This helps
% improve the convergence of training.

layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%% Train CNN Using CIFAR-10 Data
% Now that the network architecture is defined, it can be trained using the
% CIFAR-10 training data. First, set up the network training algorithm
% using the |trainingOptions| function. The network training algorithm uses
% Stochastic Gradient Descent with Momentum (SGDM) with an initial learning
% rate of 0.001. During training, the initial learning rate is reduced
% every 8 epochs (1 epoch is defined as one complete pass through the
% entire training data set). The training algorithm is run for 40 epochs.
%
% Note that the training algorithm uses a mini-batch size of 128 images. If
% using a GPU for training, this size may need to be lowered due to memory
% constraints on the GPU.

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'Verbose', true); 


doTraining = true;

if doTraining    
    % Train a network.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
    % Load pre-trained detector for the example.
    load('rcnnStopSigns.mat','cifar10Net')       
end

%% Validate CIFAR-10 Network Training
% After the network is trained, it should be validated to ensure that
% training was successful. First, a quick visualization of the first
% convolutional layer's filter weights can help identify any immediate
% issues with training. 

% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale and resize the weights for better visualization
w = mat2gray(w);
w = imresize(w, [100 100]);

figure
montage(w)



% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)


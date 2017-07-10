close all;
clear all;
clc

load ResizeARDatabase_540;
% TrainFace is the training data, each column is one sample
% TestFace is the testing data, each column is one sample
% TrainLabel gives the identity information of training data
% TestLabel gives the identity information of testing data

%number of testing samples
teNum=size(TestLabel,2);
%number of training classes
trClass=max(TrainLabel);

% normalize each sample with unit l_2 norm
D = TrainFace./ repmat(sqrt(sum(TrainFace.*TrainFace)),[size(TrainFace,1) 1]);
test = TestFace./ repmat(sqrt(sum(TestFace.*TestFace)),[size(TestFace,1) 1]);

% parameters setting for slr
v = 1*sqrt(size(TrainFace,1))^-1;
lambda = 1;
tau = 0.01;
delta = 1.2 * v;
eta = 1 * v;


            
% slr procedure
[A, B, X, E] = slr_iteration(D, TrainLabel, lambda, tau, eta, delta, 1e-4, 1000, 4);

% parameters setting for sdr
beta=10;
gamma=10;

L=[];

for i =1:teNum
    fprintf('Recognizing %d  out of %d query image\n', i, teNum);
    
    d2=[];
    
    % get the ith testing image
    y = test(:,i);
    
    % sdr procedure
    [a x e]=sdr_ialm(A, B, y, beta, gamma, 5e-3, 1000);
    
    % recovered clean face
    dny=y-B*x-e;

    % class residual
    for k = 1:trClass
        residual = dny - A(:,k==TrainLabel) * a(k==TrainLabel);
        d2(k) = residual' * residual;
    end
    % classification result
    [r L(i)] = min(d2);
end
accuracy = sum(L==TestLabel) / teNum;
fprintf('The recogniiton rate is %f', accuracy);
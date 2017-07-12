function s = SetParams
% SetParams set the parameters used throughout the package
%
% Please cite the following paper if you use the code:
%
% Peihua Li,  Qilong Wang, Wangmeng Zuo, and Lei Zhang. Log-Euclidean Kernels for Sparse 
% Representation and Dictionary Learning. IEEE Int. Conf. on Computer Vision (ICCV), 2013.
%
% For questions,  please conact:  Qilong Wang  (Email:  wangqilong.415@163.com), 
%                                               Peihua  Li (Email: peihuali at dlut dot edu dot cn) 
%
% The software is provided ''as is'' and without warranty of any kind,
% experess, implied or otherwise, including without limitation, any
% warranty of merchantability or fitness for a particular purpose.

TextureKinds = 200;
TrainNum = 3;
TestNum = 4;
d = 43;

TrainData = zeros(d,d,TrainNum*TextureKinds);
TrainPos = 1:TrainNum:TrainNum*TextureKinds;

s = struct();
% The types of kernel:  
% 'Log-E poly.' -- Log-Euclidean polynomial kernel
% 'Log-E exp.' -- Log-Euclidean exponential kernel
% 'Log-E Gauss.' --Log-Euclidean Gaussian kernel
if (isfield(s, 'kernel') == 0),
    s.kernel = 'Log-E poly.' ; 
end
% The degree of monomial in polynomial kernel or exponential kernel
if (isfield(s, 'n') == 0),
    s.n = 50;
end
% The parameter $\beta$ in the Gaussian kernel
if (isfield(s, 'Beta') == 0),
    s.Beta = 2e-2;
end
% The regularizing coefficient in the objective of sparse coding
if (isfield(s, 'SR_Lambda') == 0),
    s.SR_Lambda = 1e-3;
end
% Type of features
%  'ori'-The original covariance matrix;
%  'log' The logarithms of the  covariance matrix
if (isfield(s, 'features') == 0),
    s.features = 'log';
end
if (isfield(s, 'TrainNum') == 0),
    s.TrainNum = TrainNum;
end
if (isfield(s, 'TestNum') == 0),
    s.TestNum = TestNum;
end
if (isfield(s, 'TextureKinds') == 0),
    s.TextureKinds = TextureKinds;
end
if (isfield(s, 'TrainData') == 0),
    s.TrainData = TrainData;
end
if (isfield(s, 'TrainPos') == 0),
    s.TrainPos = TrainPos;
end

end
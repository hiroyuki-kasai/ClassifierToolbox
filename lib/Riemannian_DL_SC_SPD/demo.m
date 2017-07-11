% Riemannian Dictionary Learning and Sparse Coding 
%  if you plan to use this code, please cite 
% Cherian, Anoop, and Suvrit Sra. "Riemannian Dictionary Learning and Sparse Coding for Positive Definite Matrices." arXiv preprint arXiv:1507.02772 (2015).
% This code is released on the BSD3 license.
% This code should not be used for non-commercial purposes. 
% The authors are not liable to any loss or damage caused by running this code.
% For any issues/bugs, please contact anoop.cherian@gmail.com.

% Copyright (c) 2016, Anoop Cherian
% All rights reserved.
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
% OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

addpath ./spalgos/;
addpath(genpath('./manopt/'));
addpath ./tools;
key = 1234;
randn('state', key); rand('twister', key);

n=100; num_atoms=20; active_size=5; d=5;
[data, B, A] = generate_data(n, d, num_atoms, active_size); data=data';
data = data(randperm(n));

% sparsity for varying lambda:
lambda = 0.1;
[BB, alpha, obj] = Fast_Riem_DL(data, num_atoms, active_size, 'sim', lambda, 1234);

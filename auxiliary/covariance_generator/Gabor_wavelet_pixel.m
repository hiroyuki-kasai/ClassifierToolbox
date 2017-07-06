function gw = Gabor_wavelet_pixel(m, n, Kmax, f, u, v, Delta2)
% Create the Gabor Wavelet Filter
% Author : Chai Zhi  
% e-mail : zh_chai@yahoo.cn

% Modified by H.Kasai on June 22, 2017
%
% References:
%       Yanwei Pang, Yuan Yuan, and Xuelong Li 
%       'Gabor-Based Region Covariance Matrices for Face Recognition'
%       IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
%
% Input
%       Kmax        maximum frequency (usualy set to 5 in face recognition)
%       Delta2       \sigma^2 (= 2*pi)
%
% Outpu
%       gw          Scalor of Gabor wavelet at (m,n).



%k = ( Kmax / ( f ^ v ) ) * exp( i * u * pi / 8 );% Wave Vector
k = ( Kmax / ( f ^ v ) ) * exp( 1i * u * pi / 8 );% Wave Vector  %HK

kn2 = ( abs( k ) ) ^ 2; % \| k_{u,v} \|^2 in Eq.(12).

% Gabol wavelet \varphi_{u,v}(x,y) in Eq.(12)
gw = ( kn2 / Delta2 ) * exp( -0.5 * kn2 * ( m ^ 2 + n ^ 2 ) / Delta2) * ( exp( 1i * ( real( k ) * m + imag ( k ) * n ) ) - exp ( -0.5 * Delta2 ) );
    

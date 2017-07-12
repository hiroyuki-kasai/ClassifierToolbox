function [opts] = IPM_SC ( ipts , par)
% ========================================================================
% Spaese Coding of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------- 
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm to do sparse coding
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang,"Fisher Discrimination 
% Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on
% Computer Vision, 2011.
% L. Rosasco, A. Verri, M. Santoro, S. Mosci, and S. Villa. Iterative
% Projection Methods for Structured Sparsity Regularization. MIT Technical
% Reports, MIT-CSAIL-TR-2009-050,CBCL-282, 2009.
% J. Bioucas-Dias, M. Figueiredo, ?A new TwIST: two-step iterative shrinkage
% /thresholding  algorithms for image restoration?, IEEE Transactions on 
% Image Processing, December 2007.
%----------------------------------------------------------------------
%
%  Inputs :   (1) ipts :    the structre of input data
%                    .D     the dictionary
%                    .tau1    parameter of l1-norm energy of coefficient
%                    .y     the testing sample
%             (2) par :     the struture of input parameters
%                    .eigenv   
%
% Outputs:    (1) opts :    the structure of output data
%                    .A     the coefficient matrix
%                    .ert   the total energy sequence
%
%---------------------------------------------------------------------

ipts.cT         =      1e+10;  % stop criterion
ipts.citeT      =      1e-6;   % stop criterion
par.nIter       =      200;    % maximal iteration number
par.cRatio      =      1.05;   
ipts.initM      =      'zero'; % coefficiet initialization method
    
switch lower(ipts.initM)
    case {'zero'}
        x(:,1)  =  zeros(size(ipts.D,2),1);
    case {'transpose'}
        x(:,1)  =  ipts.D'*ipts.y;
    case {'pinv'}
        x(:,1)  =  pinv(ipts.D)*ipts.y;
    otherwise
        error('Nonknown method!');
end

A         =         ipts.D;
tau1      =         ipts.tau1;
y         =         ipts.y;
nIter     =         par.nIter;
if ~isfield(par,{'eigenv'})
c         =         par.cRatio*find_max_eigenv(A'*A);
else
c         =         par.cRatio*par.eigenv ;
end

%%%%%%%%%%%%%%%%%%%%
% TWIST parameter
%%%%%%%%%%%%%%%%%%%%
for_ever           =         1;
IST_iters          =         0;
TwIST_iters        =         0;
sparse_sign        =         1;
verbose            =         1;
enforceMonotone    =         1;
lam1               =         1e-4;   %default eigenvalues
lamN               =         1;      %default eigenvalues
rho0               =         (1-lam1/lamN)/(1+lam1/lamN); 
alpha              =         2/(1+sqrt(1-rho0^2));        %default,user can set
beta               =         alpha*2/(lam1+lamN);         %default,user can set

%%%%%%%%%%%%%%%%%%%
%main loop
%%%%%%%%%%%%%%%%%%%
xm2       =      x(:,1);
xm1       =      x(:,1);
prev_f    =      norm(y-A*x(:,1),2)^2+...
                    tau1*norm(x(:,1),1);  
for n_it = 2 : nIter;

   ert(n_it-1)      =       norm(y-A*x(:,n_it-1),2)^2+...
        tau1*norm(x(:,n_it-1),1);
   fprintf('Iteration:%f  Total gap:%f\n',n_it,ert(n_it-1));
    
    while for_ever
       % IST(IPM) estimate
       v        =       1/c*A'*(y-A*xm1)+xm1;
       x_temp   =      soft(v,tau1/c/2);

       if (IST_iters >= 2) | ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse_sign
                mask    =   (x_temp ~= 0);
                xm1     =   xm1.* mask;
                xm2     =   xm2.* mask;
            end
            % two-step iteration
            xm2    =   (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x_temp;
            % compute residual
            f      =   norm(y-A*xm2,2)^2+...
                       tau1*norm(xm2,1);
            if (f > prev_f) & (enforceMonotone)
                TwIST_iters   =  0;  % do a IST(IPM) iteration if monotonocity fails
            else
                TwIST_iters =   TwIST_iters+1; % TwIST iterations
                IST_iters   =    0;
                x_temp      =   xm2;
                if mod(TwIST_iters,10000) ==0
                   c = 0.9*c; 
                end
                break;  % break loop while
            end
        else
            f     =     norm(y-A*x_temp,2)^2+...
                        tau1*norm(x_temp,1);
            if f > prev_f
                % if monotonicity  fails here  is  because
                % max eig (A'A) > 1. Thus, we increase our guess
                % of max_svs
                c         =    2*c;        
                if verbose
%                     fprintf('Incrementing c=%2.2e\n',c);
                end
                if  c > ipts.cT
                    break;  % break loop while    
                end
                IST_iters = 0;
                TwIST_iters = 0;
            else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
            end
        end
    end

    citerion      =   abs(f-prev_f)/prev_f;
    if citerion < ipts.citeT | c > ipts.cT
%        fprintf('Stop!\n c=%2.2e\n citerion=%2.2e\n',c,citerion);
       break;
    end
    
    xm2           =   xm1;
    xm1           =   x_temp;
    x(:,n_it)   =   x_temp;
    prev_f        =   f;
end

opts.x     =       x(:,end);
opts.ert   =       ert;
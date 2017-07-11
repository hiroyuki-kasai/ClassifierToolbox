function out = matrixbbnnls(P, A, b, x0, lambda, opt)
% BBNNLS   -- Solve Matrix variant of NNLS problems via SBB
% arguments:
% A = dictionary v x d, 
% P = A'*A for avoiding recomputation of this thing.
% b = data matrix. v x v.
% x0= initial estimate
% lambda = regularization.
%
% WARNING Use at own risk!
% NOTE --- guaranteed convergence phase: *REMOVED* for speedup!!
% 
% function out = bbnnls(A, b, x0, opt)
% Solve a bound-constrained least squares problem, 
%    min    0.5*||AxA'-b||_F^2 + \lambda*x, s.t. x >= 0
%    for diag x.
%
% x0 -- Starting vector (useful for warm-starts).
%
% OPT -- This structure contains important opt that control how the
% optimization procedure runs. To obtain a default structure the user can
% use 'opt = solopt'. Use 'help solopt' to get a description of
% what the individual opt mean.
%
% Most important options to tune as: opt.tolg, opt.maxit
%
%
% OUT contains the solution and other information about the optimization or
% info indicating whether the method succeeded or failed.
%
% See also: solopt, bcls
%
% Version 1.1 (c) 2010 Suvrit Sra, Dongmin Kim
% if you use this code, please cite this paper:
%  "A non-monotonic method for large-scale non-negative least squares"
%    by D. Kim, S. Sra, I. S. Dhillon, Optimization Methods and Software, 2011.
%
% Version 1.2 (c) Suvrit Sra, Anoop Cherian, 
%    Modified by Anoop for Generalized Dictionary Learning , ECML 2011.
%

    fgx = @(x) myfuncGrad(P, A, b, x, lambda); % function to compute obj and grad

    % do some initialization for maintaining statistics
    out.iter = 0;
    out.iterTimes = nan*ones(opt.maxit,1);
    out.objTimes  = nan*ones(opt.maxit,1);
    out.pgTimes   = nan*ones(opt.maxit,1);
    out.trueError = nan*ones(opt.maxit,1);
    out.startTime = clock;
    out.status = 'Failure';

    % HINT: Very important for overall speed is to have a good x0
    out.x      = x0;
    out.refx   = x0;
    [out.refobj, out.grad]   = fgx(out.x);
    out.oldg   = out.grad;
    out.refg   = out.oldg;


    %% Begin the main algorithm
%     fprintf('Running: **** SBB-NNLS ****\n\n');
%     fprintf('Iter   \t     Obj\t\t  ||pg||_inf\t\t ||x-x*||\n');
%     fprintf('-------------------------------------------------------\n');

    while 1
        out.iter = out.iter + 1;

        % HINT: edit checkTermination to determine good criterion for yourself!
        [termReason, out.pgTimes(out.iter)] = checkTermination(opt, out);
        if (termReason > 0), break; end

        % HINT: computeBBStep is the one to implement most carefully
        [step out] = computeBBStep(P, out);% Anoop - computeBBStep(A, b, out);
        out.x = out.x - step * out.grad;
        out.oldg = out.grad;
        
        % HINT: projection step: can replace the 0 by an epsilon to truncate
        % values close to 0
        out.x(out.x < 0) = 0;

        [out.obj out.grad] = fgx(out.x);
        
        % HINT: can remove, as this is just for statistics
        out.objTimes (out.iter) = out.obj;
        out.iterTimes(out.iter) = etime(clock,out.startTime);
        
        % HINT: for debugging, to see how result develops if true x* is known
        if (opt.truex), out.trueError(out.iter) = norm(opt.xt-out.x); end
        if (opt.verbose)
            fprintf('%04d\t %E\t%E\t%E\n', out.iter, out.obj, out.pgTimes(out.iter), out.trueError(out.iter)); 
        end
    end % of while

    %%  Final statistics and wrap up
    out.time = etime(clock, out.startTime);
    out.status = 'Success';
    out.termReason = setTermReason(termReason);
end

% Compute BB step; for SBB also modifies out.oldg, and this change must be
% passed back to the calling routine, else it will fail!
function [step out] = computeBBStep(P, out) 
    
    % HINT: Can tune the x==0 to replace it by an epsilon to do TUNING
    gp = find(out.x == 0 & out.grad > 0);
    out.oldg(gp) = 0;

    Pg = P*out.oldg; % Anoop. Note that gradient is now \nabla f=P.diag(x).P-A'.b'.A + \lambda I. where P=A'.A
    
    % HINT: In my experience, the falling alternating steps perform better
    if (mod(out.iter, 2) == 0)
        step = (out.oldg' * out.oldg) / trace(Pg' * Pg); % Anoop: step is now \|g'g\|^2/\|Pg'.Pg\|^2_F
    else
        numer = Pg' * Pg;
        Pg = P*Pg; % 
        Pg(gp) = 0;
        step = numer / trace(Pg' * Pg);
    end
    step=diag(diag(step));
end

% function to return the objective for the gdl problem.
% min_{x>=0} 0.5 norm(b-AxA')^2 + \lambda x
function [f g] = myfuncGrad(P, A, b, x, lambda)
    f = 0.5*norm(b-A*x*A','fro')^2 + lambda*sum(sum(x));   
    if (nargout > 1)
        g = diag(diag(P*x*P - A'*b*A + lambda*eye(size(A,2))));
    end
end

% check various termination criteria; return norm of pg
% the strictest is norm of pg
% HINT: for speedup, use maybe just opt.tolo or some other criterion that
% you like.
function [v pg] = checkTermination(options, out)
    % pgnorm limit -- need to check this first of all
    gp = find(out.x ~= 0 | out.grad < 0);

    pg = norm(out.grad(gp), 'inf');
    if (pg < options.tolg), v=8; return; end

    % First check if we are doing termination based on running time
    if (options.time_limit)
        out.time = etime(clock, out.start_time);
        if (out.time >= options.maxtime)
            v = 1;
            return;
        end
    end

    % Now check if we are doing break by tolx
    if (options.use_tolx)
        if (norm(out.x-out.oldx, 'fro')/norm(out.oldx, 'fro') < options.tolx)
            v = 2;
            return;
        end
    end

    % Are we doing break by tolo (tol obj val)
    if (options.use_tolo && out.iter > 2)
        delta = abs(out.objTimes(out.iter-1)-out.objTimes(out.iter-2));
        if (delta < options.tolo)
            v = 3;
            return;
        end
    end

    % Finally the plain old check if max iter has been achieved
    if (out.iter >= options.maxit)
        v = 4;
        return;
    end

    % KKT violation
    if (options.use_kkt)
        if abs(out.x' * out.grad) <= options.tolk
            v = 7;
            return;
        end
    end


    % All is ok...
    v = 0;
end

%% Prints status
function showStatus(out, options)
    if (options.verbose)
        fprintf('.');
        if (mod(out.iter, 30) == 0)
            fprintf('\n');
        end
    end
end

% String representation of termination
function r = setTermReason(t)
    switch t
      case 1
        r = 'Exceeded time limit';
      case 2
        r = 'Relative change in x small enough';
      case 3
        r = 'Relative change in objvalue small enough';
      case 4
        r = 'Maximum number of iterations reached';
      case 5
        r = '|x_t+1 - x_t|=0 or |grad_t+1 - grad_t| < 1e-9';
      case 6
        r = 'Line search faild';
      case 7
        r = '|x^T * grad| < opt.pbb_gradient_norm';
      case 8
        r = '|| grad ||_inf < opt.tolg';
      case 100
        r = 'The active set converged';
      otherwise
        r = 'Undefined';
    end
end


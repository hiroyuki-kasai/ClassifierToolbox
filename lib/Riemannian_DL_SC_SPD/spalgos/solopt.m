function options = solopt(varargin)
% SOLOPT  --  Creates a default options structure
%
% OPTIONS = SOLOPT
%

options.asgui = 0;
options.beta = 0.0498;
options.compute_obj = 1;
% diminishing scalar; beta^0 =  opt.dimbeg
% beta^k = opt.dimbeg / k^opt.dimexp
options.dimexp = .5;
options.dimbeg = 5;
options.maxit = 100;
options.maxtime = 10;
options.maxnull = 10;
options.max_func_evals = 30;
options.pbb_gradient_norm = 1e-9;
options.sigma = 0.298;
options.step  = 1e-4;
options.tau = 1e-7;             
options.time_limit = 0;
options.tolg = 1e-3;
options.tolx = 1e-8;
options.tolo = 1e-5;
options.truex=0;
options.use_kkt = 0;
options.use_tolg = 1;
options.use_tolo = 0;
options.use_tolx = 0;
options.useTwo = 0;
options.verbose = 1;                    % initially
if nargin == 1
  options.variant = varargin{1};
else   % Default
  options.variant = 'SBB';
end


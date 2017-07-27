function out=sdivLasso(B,X,lam,mit,algo,gamma)
   
   switch algo
     case 0
       gfx  = @(x) fungradFrob(x,B,X,lam);
     case 1
       gfx  = @(x) fungradRiem(x,B,X,lam);
   end
   
   prx  = @(x) proj(x);
   x0   = ones(numel(B),1); x0 = x0/sum(x0);
   options.verbose = 0;
   options.testOpt = 1;
   [out.x,out.f,out.funEvals,out.projects,out.itertime,out.info] = cleanSPG(gfx,x0,prx,mit,options);
end


function [f,g]=fungradFrob(x,B,X,lam)
   n=size(X);
   Bx=am(B,x);
   f = 0.5*norm(Bx-X,'fro')^2+lam*sum(x);
   if (nargout > 1)
      g = zeros(numel(B),1);
      for i=1:numel(B)
		g(i) = vec(B{i})'*vec(Bx-X)  + lam;
      end
   end
end

function [f,g] = fungradRiem(x,B,X,lam)    
    n = numel(B);   small = eye(size(B{1},1))*1e-6;     
    Bx = am(B,x);    
    f = 0.5*riem(Bx,X)^2 + lam*sum(x);    
    lgpsdi = logpsd(X\Bx)/(Bx+small);
    g = zeros(n,1);    
    for i=1:n
        g(i) = trace(lgpsdi * B{i}); 
    end    
    g = g + lam;
end

function x=proj(y)
   x=y;
   x(x<0)=0;   
   % we will project the data to the l1 norm given by 0.5;
   %if norm(x,1)>0.5
	%x = 0.5 *x/norm(x,1);
   %end
end

function Y=Balpha(x,B)
   Y=x(1)*B{1};
   for i=2:numel(B)
      Y = Y + x(i)*B{i};
   end
end



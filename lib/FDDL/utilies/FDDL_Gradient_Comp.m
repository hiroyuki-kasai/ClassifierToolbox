function grad = FDDL_Gradient_Comp(Xi,Xa,classn,index,lambda2,lambda3,lambda4,tau2,...
    tau3,trls,drls,newpar,BAI,CJ)
% ========================================================================
% IPM's Gradient computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Xi:   the coefficient matrix of this class
%           (2) Xa:   the coefficient matrix of the whole class
%           (3) classn:   the number of class
%           (4) index: label of class being processed
%           (5) lambda2 : parameter of within-class scatter
%           (6) lambda3 : parameter of between-class scatter
%           (7) lambda4:  parameter of l2-norm energy of coefficient
%           (8) tau2: parameter of ||A_i-D_iX_i^i||_F^2 in fidelity term
%           (9) tau3: parameter of ||D_jX_i^j||_F^2 in  fidelity term
%           (10) trls: labels of training samples
%           (11) drls: labels of dictionary's column
%           (12) newpar, BAI, and CJ: the precomputed data        
%
% Outputs : (1) grad  :    the gradient vector of coding model
%
%------------------------------------------------------------------------

n_d             =      newpar.n_d;                % the sample number of i-th training data
B_line_i        =      newpar.B_line_i;
C_j             =      newpar.C_j;
C_line          =      newpar.C_line;
DD              =      newpar.DD;
DAi             =      newpar.DAi;
Di0Di0          =      newpar.Di0Di0;
Di0Ai           =      newpar.Di0Ai;
BiBi            =      newpar.BiBi;
BaiBai          =      newpar.BaiBai;
BaiGxi          =      newpar.BaiGxi;
CjCj            =      newpar.CjCj;
m               =      newpar.m;
DoiDoi          =      newpar.DoiDoi;
B_angle_i       =      newpar.Bai;

for k = 1:classn
%     Z(k).Matrix   =   Xa(:,trls==k)*B_line_i-Xa*C_line+Xi*C_j;
%     Z(k).Matrix   =   Xa(:,trls==k)*B_angle_i-Xa*C_line+Xi*C_j+Xa(:,trls==k)*C_j;
    Z(k).Matrix   =   Xa(:,trls==k)*BAI(k).M-Xa*C_line+Xi*C_j+Xa(:,trls==k)*CJ(k).M;
end

XiT      =   Xi';

tem      =   2*DD*Xi-2*DAi;
grad1    =   tem(:);

tem      =   tau3*2*DoiDoi*Xi+(tau2)*2*Di0Di0*Xi-(tau2)*2*Di0Ai;
grad7    =   tem(:);

% grad8 = zeros(size(grad7));
% for i =1:classn
%     Di0 = D;
%     Di0(:,drls~=i)=0;
%     tem = -(tau3*2*Di0'*Di0*Xi-tau3*2*Di0'*Ai);
%     grad8  =  grad8+tem(:);
% end

tem       =   2*lambda2*BiBi*XiT;
grad4     =   tem(:);

tem       =  2*lambda4*XiT;
grad9     =  tem(:);

tem       =  -lambda3*(2*BaiBai*XiT-2*BaiGxi);
grad5     = tem(:);

grad6 = zeros(size(grad5));
for k = 1:classn
    temz  =  Z(k).Matrix;
    if k~=index
        tem = -lambda3*(2*CjCj*XiT-2*C_j*temz');
        grad6 = grad6+tem(:);
    end
end

grad456 = reshape(grad4+grad5+grad6+grad9,[n_d m])';
grad456 = grad456(:);

% grad = grad1+grad456+grad7+grad8;
grad = grad1+grad456+grad7;

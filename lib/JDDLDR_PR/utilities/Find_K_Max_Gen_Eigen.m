%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Eigen_Vector,Eigen_Value]=Find_K_Max_Gen_Eigen(Matrix1,Matrix2,Eigen_NUM)

[NN,NN]=size(Matrix1);
[V,S]=eig(Matrix1,Matrix2); %Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %

S=diag(S);
[S,index]=sort(S);

Eigen_Vector=zeros(NN,Eigen_NUM);
Eigen_Value=zeros(1,Eigen_NUM);

p=NN;
for t=1:Eigen_NUM
    Eigen_Vector(:,t)=V(:,index(p));
    Eigen_Value(t)=S(p);
    p=p-1;
end
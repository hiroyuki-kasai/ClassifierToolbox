
function [D,C]=JDDLDR_UDC(X,trls,lambda_a,lambda_b,D,C,Max_iteration)

nClass = max(trls);
Jstep_T = 1e-3;

disp_cycle = floor(nClass/10);
if disp_cycle < 1
    disp_cycle = 1;
end

fprintf('Update of D and C: class ');
for ci = 1:nClass
    cdat = X(:,trls==ci);
    temD = D(ci).M;
    iteration = 1;
    while  iteration < Max_iteration
    % update C
    afa = inv(temD'*temD+lambda_a*eye(size(temD,2)))*(temD'*cdat);
    PinvD = inv(temD'*temD+(lambda_a+lambda_b)*eye(size(temD,2)));
    for tj = 1:10
%         plot(afa(:,1));title(num2str(tj));pause(1);
        avg_afa = repmat(mean(afa,2),[1 size(afa,2)]);
        afa = PinvD*(temD'*cdat+lambda_b*avg_afa);
    end
    C(ci).M = afa;
    
    % update D
    for i=1:size(temD,2)
        ai        =    afa(i,:);
        Y         =    cdat-temD*afa+temD(:,i)*ai;
        di        =    Y*ai';
        di        =    di./norm(di,2);
        temD(:,i)    =    di;
    end
    D(ci).M  = temD;
    
    zz            =    cdat-temD*afa;
    zalpha        =    afa(:);
    avg_afa       =    repmat(mean(C(ci).M,2),[1 size(C(ci).M,2)]);
    z_afa         =    afa - avg_afa;
    Jnow          =    zz(:)'*zz(:)+lambda_a*sum(zalpha(:).*zalpha(:))+lambda_b*sum(z_afa(:).*z_afa(:));
    iteration     =    iteration+1;
    end
    
    if mod(ci, disp_cycle) == 0
        fprintf('%03d ', ci);
    end  
end
fprintf('\n');

%%Flexible structure with consensus information for multi-view feature selection
function [Hv,obj]=CDMvFS(fea,alpha,beta,gamma,n,v,c)
[v1,v2]=size(fea);
Hv=cell(v1,v2);
Sv=cell(v1,v2);
Dv=cell(v1,v2);
Swsum=cell(v1,v2);
P=ones(v2,v2);
Q=ones(v2,1);
G=ones(v2,1);
d=zeros(v2,1);
w=zeros(v2,1);
Aeq=ones(1,v2);
Beq=1;
lb=zeros(v2,1);
MaxIter=20;

Ssum=0;
for num = 1:v
    fea{num}=fea{num}';
    d(num)=size(fea{num},1);
    Hv{num}=randn(d(num),c);
    Dv{num}=eye(d(num));
    w(num)=1/v;
    Sv{num} = constructW_PKN(fea{num}, 5, 1);
    Ssum=Ssum+Sv{num};
end
V=2*full(sqrt(mean(mean(Ssum))/c))*rand(n,c);

for iter=1:MaxIter

    %%update Sv
    for num = 1:v
        sumSw=0;
        for num2 = 1:v
            if num2==num
                continue
            end
            sumSw=sumSw+(gamma+2*w(num2))*Sv{num2};
        end
        intermedia=2*fea{num}'*Hv{num}*Hv{num}'*fea{num};
        Sv{num}=(intermedia+2*w(num)*eye(n)+2*beta*eye(n))\(intermedia+2*V*V'-sumSw);
    end
     
    %%update Hv
    for num = 1:v
        LG = (eye(n) - Sv{num});
        LG = LG * LG';
        LG = (LG + LG') / 2;
        [Y, ~, ~]=eig1(LG, c, 0);   
       %%solve ||Y-XtH||F+alpha||H||21
        Hv{num}=(fea{num}*fea{num}'+alpha*Dv{num})\(fea{num}*Y);
        Hi=sqrt(sum(Hv{num}.*Hv{num},2)+eps);
        diagonal=0.5./Hi;
        Dv{num}=diag(diagonal);
    end
    
    %%update V
    sumwS=0;
    for num = 1:v
        sumwS = sumwS +w(num)*(Sv{num}+Sv{num});
    end
    [V,~,~]=eig1(eye(n)-sumwS,c,0);

    %%update w
    for num = 1:v
        for num2 = 1:v
            Fii=Sv{num};
            Fjj=Sv{num2};
            P(num,num2) = 2*trace(Fii'*Fjj);
        end
    end
    
    for num = 1:v
        Swsum111=0;
        for num2 = 1:v
            if num2==num
                continue
            end
            Swsum111=Swsum111+gamma*trace(Sv{num}'*Sv{num2});
        end
        Swsum{num}=Swsum111;
    end
      
    for num = 1:v
        G(num)=norm(Hv{num}'*fea{num}-Hv{num}'*fea{num}*Sv{num},'fro')^2+alpha*trace(Hv{num}'*Dv{num}*Hv{num})+beta*norm(Sv{num},'fro')^2+Swsum{num};
        Q(num)=-1*G(num)+ trace(Sv{num}'*V*V')+ trace(V*V'*Sv{num});
        Q(num)=-1*Q(num);
    end
    w=quadprog(P,Q,[],[],Aeq,Beq,lb);

    sumobj=0;
    sumwS=0;
    for num = 1:v
        sumwS = sumwS +w(num)*(Sv{num});
    end
    for num = 1:v
       sumobj=sumobj+w(num)*(norm(Hv{num}'*fea{num}-Hv{num}'*fea{num}*Sv{num},'fro')^2+alpha*trace(Hv{num}'*Dv{num}*Hv{num})+beta*norm(Sv{num},'fro')^2)+norm(sumwS-V*V','fro').^2;
    end
    gammasum=0;
    for num = 1:v
        for num2 = 1:v
            if num2==num
                continue
            end
            gammasum=gammasum+w(num)*gamma*trace(Sv{num}'*Sv{num2});
        end
    end
    sumobj=sumobj+gammasum;
    obj(iter)=sumobj;
     if iter >= 2 && (abs(obj(iter)-obj(iter-1)/obj(iter))<eps)
        break;
    end
    
end

end
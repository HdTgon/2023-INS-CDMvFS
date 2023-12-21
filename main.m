clc;clear all;

addpath('C:\Users\think\Desktop\AIpaper\mymymy!\paper6all\codes666\usingdatasets');

addpath('C:\Users\think\Desktop\AIpaper\mymymy!\paper6all\codes666\util');

a='3sources';

filename=['C:\Users\think\Desktop\AIpaper\mymymy!\paper6all\codes666\results\ours\',a,'.mat'];

load(a);

[~,v]=size(fea);
n=size(fea{1},1);
class_num=size(unique(gt),1);
label=gt;
dimension=0;
for num = 1:v
    dimension=dimension+size(fea{num},2);
end

feanum=[10:10:300]; % feature dimension

ACC=[];
NMI=[];
ACCstd=[];
NMIstd=[];
Fscore=[];
Fscorestd=[];
Precision=[];
Precisionstd=[];
Recall=[];
Recallstd=[];
ARI=[];
ARIstd=[];
rankmvufs=[];
mvufsobj=[];

alpha=[1e-2,1e-1,1,1e1,1e2];
beta=2.^[3:2:11];
% beta=[1e-2,1e-1,1,1e1,1e2];
% gamma=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100];
% gamma=[0.0001, 0.001, 0.01, 0.1, 1];
gamma=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];

iter=0;

for i=1:size(alpha,2)
for o=1:size(beta,2)
for l=1:size(gamma,2)  

[P,obj]=CDMvFS(fea,alpha(i),beta(o),gamma(l),n,v,class_num);

iter=iter+1

allP=[];
X=[];
for num = 1:v
    allP=[allP;P{num}];
    X=[X,fea{num}];
end

W1 = [];
for m = 1:dimension
    W1 = [W1 norm(allP(m,:),2)];
end
%% test stage
[~,index] = sort(W1,'descend');
rankmvufs(:,iter)=index;
mvufsobj(1:size(obj,2),iter)=obj;

for j = 1:length(feanum)
acc=[];
nmi=[];
fscore=[];
precision=[];
recall=[];
ari=[];
for k = 1:20
    new_fea = X(:,index(1:feanum(j)));
    idx = kmeans(new_fea, class_num,'MaxIter',200);
    res = bestMap(label,idx);
    acc111 = length(find(label == res))/length(label); % calculate ACC 
    nmi111 = MutualInfo(label,idx); % calculate NMI
    [f111,p111,r111] = compute_f(label,idx);
    [ar111,~,~,~]=RandIndex(label,idx);
    acc=[acc;acc111];
    nmi=[nmi;nmi111];
    fscore=[fscore;f111];
    precision=[precision;p111];
    recall=[recall;r111];
    ari=[ari;ar111];
end
ACC=[ACC;sum(acc)/20];
ACCstd=[ACCstd;std(acc)];
NMI=[NMI;sum(nmi)/20];
NMIstd=[NMIstd;std(nmi)];
Fscore=[Fscore;sum(fscore)/20];
Fscorestd=[Fscorestd;std(fscore)];
Precision=[Precision;sum(precision)/20];
Precisionstd=[Precisionstd;std(precision)];
Recall=[Recall;sum(recall)/20];
Recallstd=[Recallstd;std(recall)];
ARI=[ARI;sum(ari)/20];
ARIstd=[ARIstd;std(ari)];
end

end
end
end

number=size(ACC,1);
final=zeros(number,4);
for j=1:number
    final(j,1:12)=[ACC(j,1),ACCstd(j,1),NMI(j,1),NMIstd(j,1),ARI(j,1),ARIstd(j,1),Fscore(j,1),Fscorestd(j,1),Precision(j,1),Precisionstd(j,1),Recall(j,1),Recallstd(j,1)];
end

[newvalue,endindex]=sort(ACC,'descend');

final(endindex(1:10),:);

% save(filename);

function [neighbors,idx] = kNN(X,test,k)
m=size(X,1);
distances=zeros(m,1);
for i=1:m
    distances(i)=norm(X(i,:)-test);
end
idx=[];
neighbors=[];
for i=1:k
    [nearest,I]=min(distances);
    idx=[idx I];
    neighbors=[neighbors nearest];
    distances=distances(distances~=nearest);
end
end



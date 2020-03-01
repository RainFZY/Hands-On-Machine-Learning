clc
clear
close all
k=3;m=200;
x_c1=0.5*randn(m/2,2);
x_c2=1+0.3*randn(m/2,2);
test=rand(1,2); %feed your test points
scatter(x_c1(:,1),x_c1(:,2),'b','fill');
hold on;
scatter(x_c2(:,1),x_c2(:,2),'g','fill');
hold on;
scatter(test(:,1),test(:,2),'r','fill');
hold off;
legend('class 1','class 2','test data point','Location','northwest')
data=[x_c1 ones(m/2,1);x_c2 zeros(m/2,1)];
labels=data(:,3);
X=data(:,1:2);
[neighbors,Index]=kNN(X,test,k); %returns the nearest indices and the k-smallest distances
yhat=predict(labels,Index);
fprintf('The new test point classified as: class %d ',yhat);
if yhat==1
    figure(2);
    scatter([x_c1(:,1); test(:,1)],[x_c1(:,2) ; test(:,2)],'b','fill');
    hold on;
    scatter(x_c2(:,1),x_c2(:,2),'g','fill');
    hold off;
    legend('class 1','class 2','Location','northwest')
else
    figure(2);
    scatter(x_c1(:,1),x_c1(:,2),'b','fill');
    hold on;
    scatter([x_c2(:,1); test(:,1)],[x_c2(:,2) ; test(:,2)],'g','fill');
    hold off;
    legend('class 1','class 2','Location','northwest')
end
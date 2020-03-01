mu1 = [0 0];
SIGMA1 = [1 0; 0 1];
R1 = mvnrnd(mu1, SIGMA1, 300);
mu2 = [1 2];
SIGMA2 = [1 0; 0 2];
R2 = mvnrnd(mu2, SIGMA2, 200);
plot(R1(:,1),R1(:,2),'+',R2(:,1),R2(:,2),'*');


clear; clc; close all

%The only difference between this demo and the linear one is on X
%initiation. Other changes are not affecting the results

%user input (must have a header line, or data.data won't be available)
data = importdata('D:\1.txt','\t',1);
XColumnIndexs = 1; %only use 1 column in this demo
yColumnIndex = 2;
alpha = 0.1; %0.3, 0.1, 0.03, 0.01
num_iters = 200000; %number of iterations

%initialize
X = data.data(:,1);
X = [X, X.^2, X.^3, X.^4, X.^5]; %up to 5th square in this case
y = data.data(:,yColumnIndex); %vector (1 column only) of real results
m = length(y); %# of samples
n = size(X, 2); %# of features (start with 0, but the count here start from 1)

%normalization
mu = [0, mean(X)]; %size of (1, n+1)
sigma = [1, std(X)]; %size of (1, n+1)
X = [ones(m, 1), X]; %add intercept as the 1st column, size of (m, n+1)
X_norm = (X - mu)./sigma; %normalized X

%gradient descent (** use X_norm instead of X from here and below **)
theta = zeros(1, n+1); %start with 0 so add 1 for n0 here
jList = zeros(1,num_iters);
for iter = 1:num_iters
    h = theta * X_norm'; %put test samples to get values of hypothesis function
    delta = h'-y; %compute the error (predicted - measured)
    J = 1/(2*m)*sum(delta.^2); %compute cost function before update theta
	jList(iter) = J;
    theta = theta - alpha/m * (delta'*X_norm); %compute derivative of cost function at step alpha
end

%generate predicted results
yPred = sum(X_norm.*theta,2); %use sum(data, 2) to sum rows (instead of columns)
% yPred = sum((X - mu)./sigma.*theta,2); %an alternative way (using X instead of X_norm)

%an alternative way of using normal equation
thetaNorm = pinv(X_norm'*X_norm)*X_norm'*y;
yPredNorm = sum(X_norm.*thetaNorm',2);

%plot data (only for 2D case (n=1))
subplot(1,2,1)
scatter(data.data(:,1),data.data(:,2),'.b')
xlabel('X')
ylabel('Y')
grid on
hold on
plot(data.data(:,1),yPred,'r')
plot(data.data(:,1),yPredNorm,'black')
legend('Sample','Gradient Descent','Normal')

subplot(1,2,2)
plot(jList,'r')
grid on
xlabel('iteration')
ylabel('Averaged squared error')
ylim([0,10])
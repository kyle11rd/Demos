clear; clc; close all

%user input (must have a header line, or data.data won't be available)
data = importdata('D:\1.txt','\t',1);
% data = importdata('D:\2.txt',',',1);
XColumnIndexs = 1:1; % can be a sigle column or an array of columns
yColumnIndex = 2; %1 column only
alpha = 0.1; %0.3, 0.1, 0.03, 0.01
num_iters = 50; %number of iterations

%initialize
X = data.data(:,XColumnIndexs); %matrix (may be more than 1 columns) of input data
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
thetaNorm = pinv(X'*X)*X'*y;
yPredNorm = sum(X.*thetaNorm',2);

%plot data (only for 2D case (n=1)
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
clear; close all; clc

g = 1./(1+exp(-1*z));

%no regularization
h = sigmoid(theta' * X');
J = 1/m * (-1*y'*log(h)' - (1-y)'*log(1-h)');
delta = h'-y;
grad = 1/m * (delta'*X)'; %then iterate theta = theta - alpha * grad for gradient descent

p=(X*theta >= 0); %probablity of being true/positive, or P(y=1|x;theta)

%with regularization
h = sigmoid(theta' * X');
J = 1/m * (-1*y'*log(h)' - (1-y)'*log(1-h)') + lambda/2/m*sum(theta.^2) - lambda/2/m*theta(1)^2; %take away the 1st theta
delta = h'-y;
grad = 1/m * (delta'*X)';

reg = ones(length(grad),1);
reg = reg .* (lambda/m*theta);
reg(1) = 0; %remove offset to the 1st theta
grad = grad + reg;

%regularized normal equation
L = eye(n+1);
L(1,1) = 0; %remove offset to the 1st theta
theta = pinv(X'*X+lambda*L)*X'*y;
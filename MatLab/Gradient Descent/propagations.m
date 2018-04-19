clear; close all; clc

%ex4data1.mat has 5000 characters from 0 to 9 (500 1s, 500 2s, ...)
%each character has 20x20 = 400 pixels
%size(X) = 5000 x 400
%----------------------
%The label (correct value/interpretation) is stored in y
%use 10 as 0, others are all representing themselves (e.g. 1 as 1, 5 as 5)
%size(y) = 5000 x 1
%y begins with 10s (0s), then from 1 to 9 in order
load('D:\handwriting.mat');

%% theta initialization
epsi_1 = sqrt(6/(400 + 25)); %400 pixels as input, 25 units in hidden layer as output
epsi_2 = sqrt(6/(25 + 10)); %25 units in hidden layer as input, 10 labels as output
Theta1 = rand(25, 1+400)*2*epsi_1-epsi_1; %initialized thetas has theta0 for bias unit
Theta2 = rand(10, 1+25)*2*epsi_2-epsi_2;

%% forward propagation
m = size(X, 1); %# of samples (5000)
hidden_layer_size = 25; %25 hidden units
num_labels = 10; %10 labels, from 1 to 10 (0 is represented by 10)
lambda = 0.1;

a1 = [ones(m, 1), X]; %5000x401
a2 = 1.0 ./ (1+exp(-1*Theta1 * a1')); %25x401 x 401x5000 = 25x5000
a2 = [ones(1, m); a2]; %26x5000
h = 1.0 ./ (1+exp(-1*Theta2 * a2)); %10x26 x 26x5000 = 10x5000

yk = zeros(num_labels, m); %generate yk to have y for each label
for i=1:num_labels
    yk(i,:) = (y==i);
end
%yk = 10x5000, h = 10x5000, yk.*h = 10x5000
J = 1/m*(sum(sum(-1*yk.*log(h)-(1-yk).*log(1-h)))); %sum all rows and columns

Theta1Reg = Theta1(:,2:size(Theta1,2)); %remove theta_0 from theta
Theta2Reg = Theta2(:,2:size(Theta2,2));
Reg = lambda/(2*m) * ( sum(sum(Theta1Reg.^2)) + sum(sum(Theta2Reg.^2)) );
J = J + Reg; %regularization

%% backward propagation
delta3 = h - yk;
delta2 = Theta2'*delta3 .* (a2.*(1-a2));
delta2 = delta2(2:end, :); %both Theta2 and a2 contains bias parameters at beginning and need to be take away from here

Delta2 = delta3 * a2'; %calculate the big delta (as we are not calculating
Delta1 = delta2 * a1; % samples 1 by 1, use equal sign directly)

%making theta = 0 on bias unit so later regularization won't affect bias unit
Theta2_zeroBias = [zeros(size(Theta2,1),1), Theta2(:, 2:end)];
Theta1_zeroBias = [zeros(size(Theta1,1),1), Theta1(:, 2:end)];

%apply regularization
Theta2_grad = 1/m*Delta2 + lambda/m*Theta2_zeroBias; %10x26 (same as Theta2)
Theta1_grad = 1/m*Delta1 + lambda/m*Theta1_zeroBias; %25x401 (same as Theta1)

%% general looping procedure
% for i=1:loop_num
%     J = forwardProp;
%     [Theta1_grad, Theta2_grad] = backProp;
%     Theta1 = Theta1 - alpha*Theta1_grad;
%     Theta2 = Theta2 - alpha*Theta2_grad;
% end

%% Gradient checking
%1. Manually input fixed X, y, and Thetas. Since the data is fake, we can
%   set s(l) small, e.g. 3~5. Sample # m can also be small, e.g. 3

%2. Compute backprop (only once) with the set up. Print the result out

%3. Compute gradient, e.g.:

% approxGrads = zeros(size(theta));
% perturb = zeros(size(theta)); %reserve place for purterbation on theta
% e = 1e-4; %often a good choice
% for i = 1:numel(theta) %numel(3x4 matrix) = 12. if theta = [1,2;3,4;5,6], i will loop through 1 to 6
%     % Set perturbation vector
%     perturb(i) = e; %assign a purterbation to 1 theta at a time
%     loss1 = J(theta - perturb); %call the implemented function to calculate J
%     loss2 = J(theta + perturb);
%     approxGrads(i) = (loss2 - loss1) / (2*e); %approximated gradient
%     perturb(i) = 0; %release this purterbation
% end

%% Learning Curve example flow:
%for i=1:m
%    xt = X(1:i, :);
%    yt = y(1:i, :);
%    then train a theta_i with xt and yt (instead of X and y)
%    error_train(i) = J(xt, yt, theta_i)
%    error_crossValidation(i) = J(X_cv, y_cv, theta_i)
%end


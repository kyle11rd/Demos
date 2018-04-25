clear; clc; close all
%only 1 hidden layer for this demo

%% user input
data = importdata('D:\3.txt',',',1);

numFeatures = 2; %X is an #samples x #features matrix
hiddenLayerSize = 10; %# of nodes in the hidden layer
num_iters = 1000; %number of iterations

alpha = 0.03; %0.3, 0.1, 0.03, 0.01
lambda = 0.01; %larger lambda leads to less overfit (smoother)

%% initialization
X = data.data(:,1:numFeatures); %mxn (m = #samples, n = #features)
y = data.data(:,numFeatures+1); %y is always a vector (mx1 matrix)
m = length(y);
n = numFeatures;

%% normalization --  very important for regression
muX = mean(X); %get mean of each column
sigmaX = std(X); %get stdev of each column
XNorm = (X-muX)./sigmaX;

muY = mean(y); %do the same for y as its scale can be very different from
sigmaY = std(y); %XNorm and thetas which may cause very large rounding errors
yNorm = (y-muY)./sigmaY;

%% theta initializations
epsi_1 = sqrt(6/(numFeatures + hiddenLayerSize));
epsi_2 = sqrt(6/(hiddenLayerSize + 1)); %output layer size for regression is always 1
Theta1 = rand(hiddenLayerSize, 1+numFeatures)*2*epsi_1-epsi_1;
Theta2 = rand(1, 1+hiddenLayerSize)*2*epsi_2-epsi_2;

%% gradient descent
for i=1:num_iters
    %% forward propagation
    a1 = [ones(m, 1), XNorm]; %add bias unit to Xnorm
    a2 = tanh(Theta1 * a1'); %use tanh as activation function (-1 to 1)
    a2 = [ones(1, m); a2]; %add bias unit to hidden layer
    h = Theta2 * a2; %output, don't apply tanh as we want values instead of decisions
    
    J = 1/(2*m)*(sum(sum( (h-yNorm').^2 ))); %sum all rows and columns
    
    Theta1Reg = Theta1(:,2:size(Theta1,2)); %remove theta_0 from theta
    Theta2Reg = Theta2(:,2:size(Theta2,2));
    Reg = lambda/(2*m) * ( sum(sum(Theta1Reg.^2)) + sum(sum(Theta2Reg.^2)) );
    J = J + Reg; %regularization
    
    if rem(i,10) == 0 %print out J to see if cost is decreasing
        fprintf('%d\t%f\n',i,J);
    end
    
    %% backward propagation
    delta3 = h - yNorm';
    delta2 = Theta2'*delta3 .* (1-a2.^2); %a2 already in tanh format
    delta2 = delta2(2:end, :); %both Theta2 and a2 contains bias parameters at beginning and need to be taken away from here
    
    Delta2 = delta3 * a2'; %calculate the big delta (as we are not calculating
    Delta1 = delta2 * a1; % samples 1 by 1, use equal sign directly)
    
    %making theta = 0 on bias unit so later regularization won't affect bias unit
    Theta2_zeroBias = [zeros(size(Theta2,1),1), Theta2(:, 2:end)];
    Theta1_zeroBias = [zeros(size(Theta1,1),1), Theta1(:, 2:end)];
    
    Theta2_grad = 1/m*Delta2 + lambda/m*Theta2_zeroBias; %apply regularization
    Theta1_grad = 1/m*Delta1 + lambda/m*Theta1_zeroBias;
    
    Theta1 = Theta1 - alpha*Theta1_grad; %apply updates
    Theta2 = Theta2 - alpha*Theta2_grad;
end

%% get output
a1 = [ones(m, 1), XNorm];
a2 =tanh(Theta1 * a1');
a2 = [ones(1, m); a2];
h = Theta2 * a2; %forward propagate 1 last time to get hypothesis (predicted values)

h = h*sigmaY + muY; %scale back the predictions



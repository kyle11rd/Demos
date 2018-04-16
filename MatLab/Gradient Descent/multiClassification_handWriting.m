clear; close all; clc

%handwriting.mat has 5000 characters from 0 to 9 (500 1s, 500 2s, ...)
%each character has 20x20 = 400 pixels
%size(X) = 5000 x 400
%----------------------
%The label (correct value/interpretation) is stored in y
%use 10 as 0, others are all representing themselves (e.g. 1 as 1, 5 as 5)
%size(y) = 5000 x 1
%y begins with 10s (0s), then from 1 to 9 in order
load('D:\handwriting.mat');

%% display characters
%initialize a full image which displays 100 chars per row, 5 rows per label
%20 pixels x (500 char_per_label / 5 lines) = 2000 columns
%20 pixels x 10 labels x 5 lines = 1000 rows
img = zeros(1000,2000);
for i=1:5000 %index of character
    char = zeros(20,20);
    for j=1:20 %row of each character
        for k = 1:20 %column of each character
            char(j,k) = X(i,(j-1)*20+k);
        end
    end
    topLeftRowIndx = floor(i/100)*20+1;
    topLeftColIndx = rem((i-1),100)*20+1;
    img(topLeftRowIndx:topLeftRowIndx+19,topLeftColIndx:topLeftColIndx+19) = char';
end
imagesc(img, [-1, 1]); %use -1 to 1 (grey) color scale

%% gradient descent
lambda = 0.1; %regularization parameter (to reduce overfitting)
m = size(X, 1); %5000, # of characters
n = size(X, 2); %400, # of pixels per char
X = [ones(m, 1) X]; %add x0 to the 1st column, now size(X) = 5000 x 401
alpha = 0.1;
all_theta = zeros(10,n+1); %to hold thetas of each label
yTemp = ones(500,1); %one-vs-all, so only 0 and 1s
for c = 1:10 %loop through each label, keep using 10 to represent 0
    XTemp = X(y==c,:);
    theta = zeros(n + 1, 1); %each pixel has a theta, plus theta_0
    for i=1:500 %iterate 500 times each char (change it as will if needed)
        h = theta' * XTemp'; %1x401 X 401x500 = 1x500
        h = 1.0 ./ (1.0 + exp(-h)); %change range with activation function
        J = -1/m * (yTemp'*log(h)' + (1-yTemp)'*log(1-h)') + lambda/2/m*sum(theta.^2);
        J = J - lambda/2/m*theta(1)^2; %take away the 1st theta to meet convention of regularization
        
        delta = h'-yTemp;
        gradient = 1/m * (delta'*XTemp)'; %so that theta := theta - alpha*gradient
        reg = ones(length(gradient),1); %initiate regularization to gradient
        reg = reg .* (lambda/m*theta);
        reg(1) = 0; %remove offset of the 1st theta to meet the same convention
        gradient = gradient + reg;
        
        theta = theta - alpha*gradient;
    end
    all_theta(c,:) = theta';
end

%% check the success rate
hAll = (all_theta * X')'; %5000 chars x 10 possible match
[~,cVec] = max(hAll,[],2); %get the index (equivalent to c in this case) of max h, 5000 x 1
cVec = round(cVec); %round to integer to make it compariable to y
fprintf('Correct detection rate is: %.2f\n',sum(cVec==y)/5000*100);



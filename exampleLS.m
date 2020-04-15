% https://www.mathworks.com/examples/matlab/mw/matlab-ex16577890-simple-linear-regression 
% https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
% openExample('matlab/LinearRegressionShortExample')

load accidents
y_ind = 4; x_ind = 14; 

x = hwydata(:,14); %Population of states
y = hwydata(:,4); % Accidents per state

x = hwydata(:,x_ind); 
y = hwydata(:,y_ind); 

% format long
b1 = x\y;

% Calculate the accidents per state yCalc from x using the relation. Visualize the regression by plotting the actual values y and the calculated values yCalc.


yCalc1 = b1*x;
figure(1);clf; shg
scatter(x,y)
hold on
plot(x,yCalc1)
xlabel(hwyheaders{x_ind})
ylabel(hwyheaders{y_ind})
title(sprintf('Linear Regression Relation Between %s & %s',hwyheaders{x_ind},hwyheaders{y_ind}))
grid on

% Improve the fit by including a y-intercept  in your model as . Calculate  by padding x with a column of ones and using the \ operator.

X = [ones(length(x),1) x];
b = X\y;

% Visualize the relation by plotting it on the same figure.
yCalc2 = X*b;
plot(x,yCalc2,'--')
legend('Data','Slope','Slope & Intercept','Location','best');

%% example of gradient ascent for logistic regression
% show loss surface, talk about convexity 
% show Gradient ascent iterations
% increase/decrease step size, show convergence
% step size: 0.1, 0.01, 0.001, 1, 2
% change intialization 

%% TODO 
% reduce X(:,2) close to zero to show the curvature effects
% increase number of samples

%% generate data 
clear

rng(5)

% sigmoid function 
sigma_f = @(z) 1./(1+exp(-z)); 

color_vec = [0 0 1; 1 0 0]; 
d = 2;

[xx1 xx2] = meshgrid([0:0.5:8]); 

% generate training samples
N = 500;
mu1 = -1*rand(1,2); 
mu2 = 1*rand(1,2); 
X = [randn(N/2,2)+mu1; randn(N/2,2)+mu2];

load fisheriris
X = meas(1:100,1:2);
N = size(X,1); 
y = [zeros(N/2,1); ones(N/2,1)];
% Y = species;
% labels = unique(Y);


% define loss function 
% https://en.wikipedia.org/wiki/Cross_entropy
err_f = @(z) sum(y(:).*log(z) + (1-y(:)).*log(1-z)); 


% using small values of X(:,2) will make the loss surface flat along 2nd
% axis 
% X(:,2) = 0.1; 

noise = 0.0*randn(N,1);

figure(1); clf; hold on;
h1 = scatter(X(:,1),X(:,2),'k');
title('input data distribution [x(1) x(2)]')
pause; 
delete(h1);
% scatter(X(1:N/2,1),X(1:N/2,2),'b');
% scatter(X(N/2+1:N,1),X(N/2+1:N,2),'r');
scatter3(X(:,1),X(:,2), y,50,color_vec(double(y>0.5)+1,:));
title('output data distribution y = g(x)')
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')
pause; 
view(40,60)
pause; 

% visualize different weights
w = [1; 1]; 
for ex = 1:10
    
    yh = sigma_f(X*w);
    h1 = scatter3(X(:,1),X(:,2), yh,50,color_vec(double(yh>0.5)+1,:),'filled');
    yh = sigma_f(w(1)*xx1+w(2)*xx2);
    h2 = surfc(xx1, xx2, yh,'FaceAlpha', 0.5000);    
    q1 = quiver3(0,0,0,w(1),w(2),0,'LineWidth',2, 'MaxHeadSize',2);
    q1.Color = 'black';   
    % view(40,60)
    yh = sigma_f(X*w); 
    loss = err_f(yh);
    title(sprintf('sigmoid function with weights: w(1) = %3.4g, w(2) = %3.4g; cross entropy loss = %3.4g',w(1), w(2),loss))
    set(gca,'FontSize',16,'FontWeight','bold')
    % legend('observed data','with constant term')
    drawnow;
    pause;
    w = randn(2,1);
    delete(h1); delete(h2); delete(q1);
end
 
%% gradient ascent 

% sigmoid function 
sigma_f = @(z) 1./(1+exp(-z)); 

% define loss function 
% look at cross entropy loss: https://en.wikipedia.org/wiki/Cross_entropy
err_f = @(z) sum(y(:).*log(z) + (1-y(:)).*log(1-z)); 

figure(2); clf;
subplot(121)
hold on;

scatter3(X(:,1),X(:,2), y,50,color_vec(double(y>0.5)+1,:));
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')


subplot(122); hold on
[w1 w2] = meshgrid([-10:0.5:10]);

lossSurface = nan(size(w1));
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2)
        yh = sigma_f(X*[w1(ii,jj); w2(ii,jj)]); 
        lossSurface(ii,jj) = err_f(yh); 
    end
end
surfc(w1,w2,lossSurface,'FaceAlpha',0.5);
set(gca,'FontSize',16,'FontWeight','bold')

tol = 1e5; 
thresh = 1e-3; 
maxiter = 1000;

% step size 
stepSize = 0.02; 

% initialization
wh = randn(2,1)*2; 
% wh = [-1; -1]*2;

y = y(:);

% gradient descent iteration 
for iter = 1:maxiter
    wh_old = wh; 
    
    yh = sigma_f(wh(1)*xx1+wh(2)*xx2);
    subplot(121);
    h1 = surfc(xx1, xx2, yh,'FaceAlpha', 0.5000);
    
    % compute current estimate
    yh = sigma_f(X*wh); 
    
    % compute gradient 
    gradW = X'*(yh-y);
    
    % update weights 
    wh = wh - stepSize*gradW; 

%     rho = 0.5;
%     c = 1e-4;
%     stepSize = linesearch(-err_f,-gradW,wh,rho,c);
%     theta = theta - alphak*g;
%             
    % new estimate 
    yh = sigma_f(X*wh); 
    h2 = line([X(:,1) X(:,1)]', [X(:,2) X(:,2)]', [yh(:) y(:)]','color','red');
    q1 = quiver3(0,0,0,wh(1),wh(2),0,'LineWidth',2, 'MaxHeadSize',2);
    q1.Color = 'black';   
    
    view(120,20)
    title(sprintf('Gradient descent iteration %d \n w = [%3.4g %3.4g]  objective value = %3.4g',iter, wh(1),wh(2), err_f(yh) ))
    set(gca,'FontSize',16,'FontWeight','bold')
    drawnow
    
    % figure(1);
    err = err_f(yh);
    subplot(122);
    h3 = scatter3(wh(1),wh(2),err,100,'filled','red');
    % h4 = scatter3(wh_old(1),wh_old(2),err,100,'filled','blue');
    view(-100,30)
    xlabel('w1')
    ylabel('w2');
    title('loss surface with constant term: $L(w) = \sum_i y_i \log h_w(x_i) + (1-y_i)\log (1-h_w(x))$','interpreter','latex')
    
    drawnow
    
    pause(1/60)
    
    if err > thresh
        break;
    end
    
    delete(h1); delete(h2); delete(q1);
end


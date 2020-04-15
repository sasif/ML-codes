%% example of gradient descent for lest squares
% show loss surface, talk about convexity 
% show Gradient descent iterations
% increase/decrease step size, show convergence
% step size: 0.1, 0.01, 0.5
% change intialization 

%% TODO 
% reduce X(:,2) close to zero to show the curvature effects
% increase number of samples

%% generate data 
clear

rng(0)

d = 2;

w = rand(d,1)-0.5;

[xx1 xx2] = meshgrid([-1:0.1:1]);

% generate training samples
N = 50;
X = rand(N,2)-0.5;

% using small values of X(:,2) will make the loss surface flat along 2nd
% axis 
% X(:,2) = 0.1; 

% generate noisy measurements 
noise = 0.1*randn(N,1);
y = X*w(1:2) + noise;

figure(1); clf;
hold on;
scatter3(X(:,1),X(:,2), y,'filled')
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')

% LS solution
% X = [X ones(N,1)];
wh = X\y;
fprintf('LS estimate: w(1) = %3.4g, w(2) = %3.4g \n',wh(1),wh(2));

yh = w(1)*xx1+w(2)*xx2;
surf(xx1, xx2, yh,'FaceAlpha', 0.5000);
view(60,20)
title(sprintf('true model = %3.4g x(1) + %3.4g x(2) + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g \n residual error = %3.4g',w(1), w(2), wh(1),wh(2),norm(y(:)-X*wh)^2))
set(gca,'FontSize',16,'FontWeight','bold')
% legend('observed data','with constant term')
drawnow;
% pause;
 
%% gradient descent 
figure(2); clf;
subplot(121)
hold on;

scatter3(X(:,1),X(:,2), y,'filled')
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')


subplot(122); hold on
[w1 w2] = meshgrid([-5:0.5:5]);
lossSurface = nan(size(w1));
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2)
        lossSurface(ii,jj) = sum(abs(y(:) - X*[w1(ii,jj); w2(ii,jj)]).^2);
    end
end
surfc(w1,w2,lossSurface,'FaceAlpha',0.5);
set(gca,'FontSize',16,'FontWeight','bold')

tol = 1e5; 
thresh = 1e-3; 
maxiter = 1000;

% step size 
stepSize = 0.1; 

% initialization
wh = randn(2,1)*2; 
% wh = [-1; -1]*2;

y = y(:);

% gradient descent iteration 
for iter = 1:maxiter
    wh_old = wh; 
    
    yh = wh(1)*xx1+wh(2)*xx2;
    subplot(121);
    h1 = surf(xx1, xx2, yh,'FaceAlpha', 0.5000);
    
    % compute gradient 
    gradW = X'*(X*wh-y);
    
    % update weights 
    wh = wh - stepSize*gradW; 

    % new estimate 
    yh = X*wh; 
    h2 = line([X(:,1) X(:,1)]', [X(:,2) X(:,2)]', [yh(:) y(:)]','color','red');
    
    view(60,20)
    title(sprintf('Gradient descent iteration %d \n w = [%3.4g %3.4g] residual error = %3.4g',iter, wh(1),wh(2), norm(y-yh)^2 ))
    set(gca,'FontSize',16,'FontWeight','bold')
    drawnow
    
    % figure(1);
    err = norm(y-yh).^2;
    subplot(122);
    h3 = scatter3(wh(1),wh(2),err,50,'filled','red');
    % h4 = scatter3(wh_old(1),wh_old(2),err,100,'filled','blue');
    view(-160,60)
    xlabel('w1')
    ylabel('w2');
    title('loss surface with constant term: $L(w) = \| y - X w\|_2^2$','interpreter','latex')
    
    drawnow
    
    pause(1/60)
    
    if err/norm(y) < thresh
        break;
    end
    
    delete(h1); delete(h2); 
end


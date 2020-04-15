%% example of linear models 1D
% show loss function 
% given data points, how to find the best linear fit
% viewpoint of hyperplanes defined in the signal space or parameter space
% each equation x_i^T w = y_i defines a hyperplane in the parameter space
% our solution lies at or near the intersection of those hyperplanes 

%% TODO 
% show a noise free setting
% show noisy setting 

%% generate data 
rng(0)

d = 1; 

% true slope of linear model
w = 0.5;%rand(d,1)-0.5;

% generate data points
N = 10;
X = rand(N,d)-0.5; 

xrange = [-0.5:0.1:0.5]; 

% create noisy observations 
noise = 0.1*randn(N,1); 
y = w(1)*X + noise; 

% plot data 
figure(1); clf; 
hold on;
scatter(X, y,'filled')
xlabel('x')
ylabel('y'); 
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')

% without constant term
wh = X\y;
yh = X*wh; 

fprintf('recovery without constant term: w(1) = %3.4g \n',wh(1));
fprintf('LS error = %3.4g\n', norm(y(:)-yh)); 

yrange = wh*xrange; 
plot(xrange, yrange)
title(sprintf('Model = w(1)*x + noise \n recovered weights: w(1) = %3.4g \n residual error = %3.4g', wh(1), norm(y(:)-yh)))
set(gca,'FontSize',16,'FontWeight','bold')
% legend('observed data','with constant term')
drawnow; 
% pause; 


%% change slope and show the loss surface and curve fit 
figure(2); clf;
subplot(121)
hold on;
scatter(X, y,'filled')

xlabel('x')
ylabel('y');
title('observed data'); 
set(gca,'FontSize',16,'FontWeight','bold')

subplot(122); hold on
w1 = sort([-1:0.1:1.2  wh]);
lossSurface = zeros(size(w1));
for ii = 1:length(w1)    
    lossSurface(ii) = norm(y - [w1(ii)*X]).^2;
end
plot(w1,lossSurface,'r');
xlabel('w1')
title('loss surface with constant term: $L(w) = \sum_i (w(1) x_i(1)- y_i)^2$','interpreter','latex')
set(gca,'FontSize',16,'FontWeight','bold')

for i = 1:N    
    yw = reshape(w1*X(i,:)-y(i),size(w1));
    
    subplot(121); 
    h1 = plot(X(i,:),y(i),'ro','MarkerSize',10); 
    
    wx = y(i)/X(i,:); 
    subplot(122); 
    scatter(wx, 0,'DisplayName',sprintf('w*x_{%d}=y_{%d}',i,i));
    l1 = legend;
    drawnow
    pause;%(1);
    delete(h1); 
end
delete(l1);

for inc = -1:0.1:2
    yh = inc*wh(1)*xrange;
    figure(2)
    subplot(121);
    h1 = plot(xrange, yh);
    yh = inc*wh(1)*X; 
    h2 = line([X(:) X(:)]',[yh(:) y(:)]','color','red');
    
    % title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), inc*wh(1),wh(2)))
    title(sprintf('estimated w = %3.4g  |  residual error = %3.4g',inc*wh(1), norm(y-yh)))
    set(gca,'FontSize',16,'FontWeight','bold')
    legend('observed data','linear fit','Location','best')
    axis([-0.5 0.5 -1 1])
    if (inc == 1)
        title(sprintf('OPTIMAL SOLUTION: estimated w = %3.4g  | residual error = %3.4g',inc*wh(1),norm(y-yh)))
    end
    drawnow
    
    % figure(1);
    subplot(122);
    yh = inc*wh(1)*X;
    h3 = scatter(inc*wh(1),norm(y - yh).^2,100,'filled','red');
    xlabel('w1')
    title('loss surface with constant term: $L(w) = \sum_i (w(1) x_i(1)- y_i)^2$','interpreter','latex')
    % axis([-wh(1) 2*wh(1) -wh(1) 2*wh(1) 0 10])
    drawnow
     
    if inc == 1
        pause;
    else
        pause(1/10)
    end
    if inc < 2
        delete(h1); delete(h2); delete(h3)
    end
end




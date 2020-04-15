%% example of linear models 2D
% show loss function 
% given data points, how to find the best linear fit
% viewpoint of hyperplanes defined in the signal space or parameter space
% each equation x_i^T w = y_i defines a hyperplane in the parameter space
% our solution lies at or near the intersection of those hyperplanes 

%% TODO 
% show a noise free setting
% show noisy setting 
% show LS with and without constant term


%% Generate data 
clear

rng(0)

% data dimension
d = 2;

% true weight vector (unknown) 
w = [0.5 0.5]';% rand(d,1)-0.5;

[xx1 xx2] = meshgrid([-1:0.1:1]);

% generate noisy data 
N = 10;
X = rand(N,2)-0.5;
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

% with constant term
% X = [X ones(N,1)];
wh = X\y;
fprintf('recovery with constant term: w(1) = %3.4g, w(2) = %3.4g \n',wh(1),wh(2));

yh = w(1)*xx1+w(2)*xx2;
surf(xx1, xx2, yh,'FaceAlpha', 0.5000);
view(60,20)
title(sprintf('true model = %3.4g x(1) + %3.4g x(2) + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')
% legend('observed data','with constant term')
drawnow; 
% pause;

%% show hyperplanes corresponding to each data point
figure(2); clf;
subplot(121)
hold on;

scatter3(X(:,1),X(:,2), y,'filled')
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
set(gca,'FontSize',16,'FontWeight','bold')
title('observed data');

subplot(122); hold on
[w1 w2] = meshgrid(sort([-1:0.1:1.2 wh(1) wh(2)]));
lossSurface = nan(size(w1));
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2)
        lossSurface(ii,jj) = norm(y(:) - X*[w1(ii,jj); w2(ii,jj)]).^2;
    end
end
surfc(w1,w2,lossSurface,'FaceAlpha',0.5);
ylabel('w2');
title('loss surface with constant term: $L(w) = \sum_i (w(1) x_i(1) + w(2)*x_i(2) - y_i)^2$','interpreter','latex')
set(gca,'FontSize',16,'FontWeight','bold')

for i = 1:N   
    wx1 = -1:0.01:1; 
    wx2 = (y(i)-wx1*X(i,1))/X(i,2);
    
    subplot(121);
    h1 = scatter3(X(i,1),X(i,2),y(i),100,'filled','red');
    subplot(122);
    
    h2 = scatter(wx1,wx2,'filled');
    view(-160,40)
    
    pause;
    
    delete(h1);
    drawnow
end

%% change both weights
for inc1 = -2:0.2:2
    for inc2 = -1:0.5:1
        
        yh = inc1*wh(1)*xx1+inc2*wh(2)*xx2;
        subplot(121);
        h1 = surf(xx1, xx2, yh,'FaceAlpha', 0.5000);
        
        yh = inc1*wh(1)*X(:,1)+inc2*wh(2)*X(:,2);
        h2 = line([X(:,1) X(:,1)]', [X(:,2) X(:,2)]', [yh(:) y(:)]','color','red');
        
        view(65,10)
        title(sprintf('estimated w= [%3.4g %3.4g] | residual error = %3.4g',inc1*wh(1),inc2*wh(2),norm(y-yh)))
        set(gca,'FontSize',16,'FontWeight','bold')
        axis([-1 1 -1 1 -2 2])
        
        if (inc1 == 1 & inc2 == 1)
            title(sprintf('OPTIMAL SOLUTION: estimated w= [%3.4g %3.4g] | residual error = %3.4g',inc1*wh(1),inc2*wh(2),norm(y-yh)))
        end
        drawnow        
        
        % figure(1);
        subplot(122);
        h3 = scatter3(inc1*wh(1),inc2*wh(2),norm(y(:) - X*[inc1*wh(1); inc2*wh(2)]).^2,100,'filled','red');
        view(-160,40)
        xlabel('w1')
        ylabel('w2');
        title('loss surface with constant term: $L(w) = \sum_i (w(1) x_i(1) + w(2)*x_i(2) - y_i)^2$','interpreter','latex')
        drawnow
        
        if inc1 == 1 & inc2 == 1
            pause;
        else
            pause(1/10)
        end
        
        if inc1 < 2
            delete(h1); delete(h2); delete(h3)
        end
    end
end

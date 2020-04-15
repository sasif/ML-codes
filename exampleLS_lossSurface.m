% example of LS loss surface with and without the bias term for 1D
% and without bias term for 2D

%% 1D data 
% solution without bias term is equivalent to setting 
% w(2) to zero or x_i(2) to zero 

rng(0)

% dimension of data
d = 1; 

% weight vector (unknown)
w = rand(d+1,1)-0.5;

% set up 
xrange = [-0.5:0.1:0.5]; 
N = length(xrange);

% create noisy measurements 
noise = 0.01*randn(1,N); 
yrange = w(1)*xrange + w(2) + noise; 

% plot data 
figure(1); clf; 
hold on;
scatter(xrange, yrange)
xlabel('x')
ylabel('y'); 
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')

% no bias/constant term
X = [xrange(:)];
y = yrange(:); 
wh = X\y;
plot(xrange, wh(1)*xrange)
fprintf('recovery without constant term: w(1) = %3.4g\n',wh);

% with constant term
X = [xrange(:) ones(N,1)];
y = yrange(:); 
wh = X\y;
fprintf('recovery without constant term: w(1) = %3.4g, w(2) = %3.4g\n',wh(1),wh(2));

plot(xrange, wh(1)*xrange+wh(2))
title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')
legend('observed data','no constant term','with constant term')

pause

%% loss surface: y-w*x (1d)
[w1 w2] = meshgrid([-1:0.1:1]);
lossSurface = nan(size(w1));
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2);
        figure(2);
        lossSurface(ii,jj) = norm(yrange - [w1(ii,jj)*xrange]).^2;
        surfc(w1,w2,lossSurface);   
        xlabel('w1')
        ylabel('w2');
        title('loss surface without constant term: $L(w) = \sum_i (w(1) x_i(1) - y_i)^2$','interpreter','latex')
    end
    drawnow
    % pause(1/60); 
end

% view(0,0); 

pause 
%% loss surface: y-w*x+b (1d + bias)
[w1 w2] = meshgrid([-1:0.2:1]);
lossSurface = nan(size(w1));
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2);
        h3 = figure(3);
        lossSurface(ii,jj) = norm(yrange - [w1(ii,jj)*xrange+w2(ii,jj)]).^2;
        h4 = surfc(w1,w2,lossSurface);   
        xlabel('w1')
        ylabel('w2');
        title('loss surface with constant term: $L(w) = \sum_i (w(1) x_i(1) + w(2) - y_i)^2$','interpreter','latex')
    end
    drawnow
    % pause(1/60); 
end

pause

%% 2D data

% dimension of data
d = 2; 

% weight vector (unknown)
w = rand(d,1)-0.5;

% set up 
[xx1, xx2] = meshgrid([-0.25:0.05:0.25]); 
N = numel(xx1);

% create noisy measurements 
noise = 0.01*randn(N,1); 
X = [xx1(:) xx2(:)]; 
yrange = X*w+ noise; 

figure(1); clf;
hold on;
scatter3(X(:,1),X(:,2), yrange,'filled')
xlabel('x(1)')
ylabel('x(2)');
zlabel('y');
% title(sprintf('true model = %3.4g x + %3.4g + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')

% without constant term
wh = X\yrange;
yh = X*wh; 
fprintf('recovery without constant term: w(1) = %3.4g, w(2) = %3.4g\n',wh(1),wh(2));

surf(xx1, xx2, reshape(yh,sqrt(N),[]),'FaceAlpha', 0.5000);
view(60,20)
title(sprintf('true model = %3.4g x(1) + %3.4g x(2) + noise \n recovered weights: w(1) = %3.4g, w(2) = %3.4g',w(1), w(2), wh(1),wh(2)))
set(gca,'FontSize',16,'FontWeight','bold')

% loss surface y = w1x1+w2x2 (2d)
[w1 w2] = meshgrid([-0.5:0.05:0.5]);
lossSurface = nan(size(w1));
% yrange = 
for ii = 1:size(w1,1)
    for jj = 1:size(w2,2);
        h3 = figure(3);
        lossSurface(ii,jj) = norm(yrange - [w1(ii,jj)*xx1(:)+w2(ii,jj)*xx2(:)]).^2;
        h4 = surfc(w1,w2,lossSurface);   
        xlabel('w1')
        ylabel('w2');
        title('loss surface without constant term: $L(w) = \sum_i (w(1) x_i(1) + w(2)x_i(2) - y_i)^2$','interpreter','latex')
    end
    drawnow
    % pause(1/60); 
end



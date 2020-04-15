%% orthogonality principle 

% here we look the space spanned by columns of X
% d columns, each of length N will define min(N,d)-dimensional subspace in
% N-dimensional space. 

% if we get any N-dimensional data vector, least-squares will project that
% vector onto the space spanned by the columns of X

clear 
rng(1)
N = 2; 
Xq = (randn(3,3)); 
% Xq = eye(3); 

X = Xq(:,1:2);
% X = [1 1; 0 1; 0 0];
Xc = Xq(:,3); 

w = randn(2,1);
noise = 0.1*randn(3,1);

y = [-0.5; 1; 0.5]; 

[w1, w2] = meshgrid([-1:0.1:1]);

figure(101); clf; hold on; 
% yw = reshape([w1(:) w2(:)]*w,size(w1)); 
% surf(w1, w2, yw);

yw = reshape(permute(X*[w1(:)'; w2(:)'],[2 1]),[size(w1) 3]); 
surf(yw(:,:,1),yw(:,:,2),yw(:,:,3))
view(30,30)

scatter3(y(1),y(2),y(3),100,'filled','black')
wh = X\y; 
yh = X*wh; 
dy = y-yh; 
scatter3(yh(1),yh(2),yh(3),100,'filled','red')
scatter3(dy(1),dy(2),dy(3),100,'filled','blue')

line([0; y(1)], [0; y(2)], [0; y(3)],'color','black')
line([0; yh(1)], [0; yh(2)], [0; yh(3)],'color','red')
line([y(1); yh(1)], [y(2); yh(2)], [y(3); yh(3)],'color','blue','LineStyle','--')
line([0; dy(1)], [0; dy(2)], [0; dy(3)],'color','blue')
line([0; X(1,1)], [0; X(2,1)], [0; X(3,1)],'color','green','LineWidth',2)
line([0; X(1,2)], [0; X(2,2)], [0; X(3,2)],'color','green','LineWidth',2)
axis equal 

X'*(y-yh)

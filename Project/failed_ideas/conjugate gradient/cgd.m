d = 50;
n = 1000;
T = 10000;
% generate the regression problem
X = [randn(n,d) ones(1000,1)];
b = randn(d+1,1);
e = 0.1;
y = X*b+randn(n,1)*e;

% original network
w = randn(d+1,T);
l = ones(T,1);
loss = @(y,y_tar) 0.5*sum((y-y_tar).^2)/numel(y_tar);
dldw = @(ww,x,y_tar) x'*(x*ww-y_tar)/numel(y_tar);
alpha = 2e-4;

% conjugate loss network
wc = randn(d+1,T);
lc = ones(T,1);
lossc = @(l,l_tar) 0.5*sum((l-l_tar).^2)/numel(l_tar);
dlcdwc = @(wc,w,l_tar) w'*(w*wc-l_tar)/numel(l_tar);
dlcdw = @(wc,w,l_tar) (w*wc-l_tar)*wc;
gamma = alpha/2;

y_est = X*w(:,1);
l(1) = loss(y_est,y);
yc_est = w(:,1:n)'*wc(:,1);
lc(1) = loss(yc_est,l(1));
for t = 1:T/n
    for i = 1:n
        % original gradient descent
        dw = -alpha*dldw(w(:,(t-1)*n+i),X(i,:),y(i));
        dw_approx = -alpha*dlcdw(wc(:,(t-1)*n+i),w(:,(t-1)*n+i)',l(i));
        w(:,(t-1)*n+i+1) = w(:,(t-1)*n+i)+dw;
        y_est = X(i,:)*w(:,(t-1)*n+i);
        l((t-1)*n+i+1) = loss(y_est,y(i));
        % conjugate loss gradient descent
        xc = w(:,1:t:(t-1)*n+i)';
        yc = l(1:t:(t-1)*n+i);
        dwc = -gamma*dlcdwc(wc(:,(t-1)*n+i),xc,yc);
        wc(:,(t-1)*n+i+1) = wc(:,(t-1)*n+i)+dwc;
        yc_est = xc*wc(:,(t-1)*n+i);
        lc((t-1)*n+i+1) = loss(yc_est,yc);
    end
end

l_est = wc(:,end)'*w;

figure;
subplot(2,2,1)
plot(w'); axis tight;
subplot(2,2,2); hold on; 
plot(1:n/2:T,mean(reshape(l(1:end-1),n/2,[])));
plot(l_est)
axis tight; 
subplot(2,2,3)
plot(wc'); axis tight;
subplot(2,2,4)
plot(mean(reshape(lc(1:end-1),n/2,[]))); axis tight; 


o = (w(:,1)'*w(:,end))/(sqrt(w(:,1)'*w(:,1))*sqrt(w(:,end)'*w(:,end)));
oc = (wc(:,1)'*wc(:,end))/(sqrt(wc(:,1)'*wc(:,1))*sqrt(wc(:,end)'*wc(:,end)));
ang = acos([o,oc])/pi*180;

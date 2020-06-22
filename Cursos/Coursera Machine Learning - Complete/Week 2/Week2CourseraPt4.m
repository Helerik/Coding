t = [0:0.01:0.99];

y1 = sin(2*pi*t);
y2 = cos(2*pi*t);

plot(t,y1, 'b');
hold on;
plot(t,y2, 'r')
xlabel('time')
ylabel('vlaue')
legend('sin', 'cos')
title('a plot')
hold off;
figure(1); plot(t, y1);
figure(2); plot(t, y2);

hold on;
subplot(1,2,1);
plot(t,y1);
axis([0,1,-1.1,1.1]);
subplot(1,2,2);
plot(t,y2);
axis([0,1,-1.1,1.1]);

A = randn(100,100);
hold off;
figure (3); imagesc(A), colorbar;
























































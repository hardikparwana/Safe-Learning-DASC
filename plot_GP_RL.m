close all;

h1 = [ 7.96 4.96 1.12 4.76 0 5.75 6.75]/( 3.0^2 - (3.3/2)^2  );

h2 = [0.95 3.94 7.79 4.15 13.16 3.16 2.16]/( (3.3/2)^2 - 0.3^2 );

h3 = [0.11 0.13 0.13 0.008 0.095 0 0.133]/( 1 - cos(pi/6) );

t = [230 138 581 245 121 900 177];

for i=1:7
    h(i) = min( [h1(i), h2(i), h3(i)] );
end

figure(1)
[h1_sorted, h1_order] = sort(h1);
t_sorted = t(h1_order);
plot(h1_sorted,t_sorted,'r-*')
% plot(h1,t,'r')
hold on

[h2_sorted, h2_order] = sort(h2);
t_sorted = t(h2_order);
plot(h2_sorted,t_sorted,'g-*')
% plot(h2,t,'g')

[h3_sorted, h3_order] = sort(h3);
t_sorted = t(h3_order);
plot(h3_sorted,t_sorted,'c-*')
% plot(h3,t,'c')


legend('Max Distance','Min Distance','Max Angle')
xlabel('Barrier function value')
ylabel('Time to learn')

figure(2)
plot(h,t,'k*')
[h_sorted, h_order] = sort(h);
t_sorted = t(h_order);


title('Barrier function vs time to learn')



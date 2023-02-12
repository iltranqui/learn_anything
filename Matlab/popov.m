function popov(sys,k,q)

%  pop with proportional gain k and popov slope q.
% notice that it is better to check the K max of the linear system by tools like rlocus, and then choose the k wisely.

% iftah naftaly (2023). pop (https://www.mathworks.com/matlabcentral/fileexchange/69734-pop), 
% MATLAB Central File Exchange. Retrieved February 12, 2023.
% drawing popov plot for given system, max k of the linear system and q -
% slope of popov line , created by Iftach Naftaly, 12.2018

figure;
[re,im,w]=nyquist(sys);
im=im(1,:).*w';
re=re(1,:);
Kmax=margin(sys);
[p z]=pzmap(sys);
str=sprintf('Popov Diagram,  K_{Min}=0   K_{Max}=%f    q=%f',Kmax,q);
for i=1:1:length(p)
    if p(i)==0
    str=sprintf('Popov Diagram,  K_{Min}= \\epsilon   K_{Max}=%f    q=%f',Kmax,q);
    end
end
plot(re,im,'b','linewidth',1.5)
hold on
if q~=0
    plot(re,q.*re+1/k,'r','linewidth',1.5);
else if q==0 
    plot((-1/k)*ones(1,length(im)),im,'r');
    end
end
hold on
plot(-1/Kmax,0,'*k')
grid on
xlabel('Re(G(i\omega))');
ylabel('\omega*Im(G(i\omega))');
set(gcf,'color','w');
legend('Non-linear function','Popov line','-1/K_{MaxLinearSys}');
set(gca,'fontsize',16)
title(str,'fontsize',18);
%% PEM Identification of Example from DDSC
close all 
clear all
%% defining the system example
sysv = tf(1e-3*[0 1 0.43 0.056 0.0023],[1 -3.1 3.5801 -1.8248 0.3462],1)

% the dataset of random variables
u = randn(5000,1);
y = lsim(sysv,u);

%% Output Error and ARX model
th_oe = oe(iddata(y,u),[2 2 1])
%nb — (Order of Numerator ) Order of the B(q) polynomial + 1, which is equivalent to the length of the B(q) polynomial. nb is an Ny-by-Nu matrix. Ny is the number of outputs and Nu is the number of inputs.
%nf — (Order of Denominator ) Order of the F polynomial. nf is an Ny-by-Nu matrix. 
%nk — Input delay ( in the lessons = d ) , expressed as the number of samples. nk is an Ny-by-Nu matrix. The delay appears as leading zeros of the B polynomial.

th_arx = arx(iddata(y,u),[2 2 1])

%% PLotting
figure; hold on;
bode(tf(th_oe.b,th_oe.f,1))
bode(tf(th_arx.b,th_arx.a,1))
title("Plot of OE and ARX")
legend('OE: Output Error','ARX')

figure;
bode(tf(th_arx.a,1,1))

figure; hold on;
bode(tf(th_oe.b,th_oe.f,1))
bode(tf(th_arx.b,th_arx.a,1))
bode(sysv)
title("Plot of OE,ARX and Original System")
legend('OE: Output Error','ARX','Original System')

figure; hold on;
bode(tf(th_oe.b,th_oe.f,1))
bode(sysv)
title("Plot of OE and Original Sys")
legend('OE: Output Error','Sys')

figure; hold on;
bode(tf(th_arx.b,th_arx.a,1))
bode(sysv)
title("Plot of ARX and Original Sys")
legend('ARX','Sys')

%% Frequency Analysis

w = [0:0.001:pi];   % pi = 3.14
hv = freqresp(sysv,w);
hoe = freqresp(tf(th_oe.b,th_oe.f,1),w);
harx = freqresp(tf(th_arx.b,th_arx.a,1),w);
figure; hold on;

% Plot of all Frequency responses
title("Frequency Response to all Models")
plot(w,vec(abs(hv)))
plot(w,vec(abs(hoe)))
plot(w,vec(abs(harx)))
legend('Sys','OE: Output Error','ARX: AutoRegressive Exogenous')

figure; hold on;
title("Frequency Response Mismatch Sys with others")
plot(w,vec(abs(hv-hoe)))
plot(w,vec(abs(hv-harx)))
legend('Sys-OE','Sys-ARX')

figure; hold on;
bode(tf(th_arx.a,[1 0 0],1))

L = tf([1 0 0],th_arx.a,1)
 
yL = lsim(L,y);
uL = lsim(L,u);

th_arxL = arx(iddata(yL,uL),[2 2 1])

% Bode ARX response
bode(L*tf(th_arxL.a,1,1))
title("ARX Bode Plot - part a")

figure; hold on;
title("ARX Bode Plot and Sys - part all")
bode(tf(th_arxL.b,th_arxL.a,1))
bode(sysv)

close all

% Requires System Toolbox Identification. 
%% Iteration OE

Fit_Best = 0;
MSE_Best = 0;

n_den = 5;
n_num = 5;
delay_max = 30;

for num=1:n_den
    for den=1:n_den
        for dist=1:delay_max
            th_oe = oe(iddata(y,u,Ts),[num,den,dist]);
            Fit = th_oe.Report.Fit.FPE;
            MSE = th_oe.Report.Fit.MSE;
            if ( Fit > Fit_Best )
                Fit_Best = FPE;
                MSE_Best = MSE;
                th_oe_best = th_oe;
            end
        end    
    end
end

th_oe_best

%% iteration ARX 

Fit_Best = 0;
MSE_Best = 0;

n_den = 5;
n_num = 5;
delay_max = 30;

for num=1:n_num
    for den=1:n_den
        for dist=1:delay_max
            th_arx = arx(iddata(y,u,Ts),[num,den,dist]);
            Fit = th_arx.Report.Fit.FPE;
            MSE = th_arx.Report.Fit.MSE;
            if ( Fit > Fit_Best )
                Fit_Best = FPE;
                MSE_Best = MSE;
                th_arx_best = th_arx;
            end
        end    
    end
end

th_arx_best

%% iteration ARMAX 
opt = armaxOptions('Focus','prediction','Display','on','EnforceStability',false,SearchMethod,'auto');
% https://de.mathworks.com/help/ident/ref/armaxoptions.html

Fit_Best = 0;
MSE_Best = 0;

n_den = 5;  % order of A polynomial ( Associated with the Output ) 
n_num = 5;  % order of B polynomial ( associated with the Input ) 
n_error = 5; %  order of C polynomial ( associated with Error ) 
delay_max = 30;

for num=1:n_num
    for den=1:n_den
        for error=1:n_error
            for dist=1:delay_max
                th_armax = armax(iddata(y,u,Ts),[num,den,error,dist],opt);
                Fit = th_armax.Report.Fit.FPE;
                MSE = th_armax.Report.Fit.MSE;
                if ( Fit > Fit_Best )
                    Fit_Best = Fit;
                    MSE_Best = MSE;
                    th_armax_best = th_armax;
                end
            end
        end    
    end
end

th_armax_best

%% iteration ARIMAX  
% ARMAX with intergration of noise 
opt = armaxOptions('Focus','prediction','Display','on','EnforceStability',false,SearchMethod,'auto');
% https://de.mathworks.com/help/ident/ref/armaxoptions.html

Fit_Best = 0;
MSE_Best = 0;

n_den = 5;  % order of A polynomial ( Associated with the Output ) 
n_num = 5;  % order of B polynomial ( associated with the Input ) 
n_error = 5; %  order of C polynomial ( associated with Error ) 
delay_max = 30;

for num=1:n_num
    for den=1:n_den
        for error=1:n_error
            for dist=1:delay_max
                th_arimax = armax(iddata(y,u,Ts),[num,den,error,dist],'IntegrateNoise',true,opt);
                Fit = th_arimax.Report.Fit.FPE;
                MSE = th_arimax.Report.Fit.MSE;
                if ( Fit > Fit_Best )
                    Fit_Best = Fit;
                    MSE_Best = MSE;
                    th_arimax_best = th_arimax;
                end
            end
        end    
    end
end

th_arimax_best
%% Final Report
report_oe = th_oe_best.Report.Fit;
report_arx = th_arx_best.Report.Fit;
report_armax = th_armax_best.Report.Fit;
report_arimax = th_arimax_best.Report.Fit;

% Print all the properties of the report
fprintf("OE best model report")
names = fieldnames(report_oe);
for i = 1:numel(names)
    fprintf('%s: %s\n', names{i}, report.(names{i}));
end
fprintf("ARX best model report")
names = fieldnames(report_arx);
for i = 1:numel(names)
    fprintf('%s: %s\n', names{i}, report.(names{i}));
end
fprintf("ARMAX best model report")
names = fieldnames(report_armax);
for i = 1:numel(names)
    fprintf('%s: %s\n', names{i}, report.(names{i}));
end
fprintf("ARIMAX best model report")
names = fieldnames(report_arimax);
for i = 1:numel(names)
    fprintf('%s: %s\n', names{i}, report.(names{i}));
end

%% Example of Displaying the best model
load dryer2;
z = iddata(y2,u2,0.08,'Tstart',0);
na = 2:4;
nc = 1:2;
nk = 0:2;
models = cell(1,18);
ct = 1;
for i = 1:3
    na_ = na(i);
    nb_ = na_;
    for j = 1:2
        nc_ = nc(j);
        for k = 1:3
            nk_ = nk(k); 
            models{ct} = armax(z,[na_ nb_ nc_ nk_]);
            ct = ct+1;
        end
    end
end

models = stack(1,models{:});
compare(z,models)
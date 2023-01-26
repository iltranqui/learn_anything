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

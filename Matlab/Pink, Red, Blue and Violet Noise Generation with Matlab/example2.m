clear, clc, close all

% signal parameters
fs = 44100;         % sampling frequency, Hz
T = 0.1;            % signal duration, s
N = round(fs*T);    % number of samples
t = (0:N-1)/fs;     % time vector

% signal generation
s = 10*sin(2*pi*100*t + pi/6);

% noise generation
SNR = 20;                       % define SNR in dB
Ps = 10*log10(std(s).^2);       % signal power, dBV^2
Pn = Ps - SNR;                  % noise power, dBV^2
Pn = 10^(Pn/10);                % noise power, V^2
sigma = sqrt(Pn);               % noise RMS, V

n = sigma*pinknoise(N);         % pink noise generation
x = s(:) + n(:);                % signal + noise mixture


% plot the signal
figure(1)
hold on;
plot(t, x, 'r', 'LineWidth', 1.5)
grid on
plot(t, s, '--b', 'LineWidth', 1.0)
xlim([0 max(t)])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Amplitude')
title('Signal + Noise in the time domain')
legend('Signal+Noise','Signal')

% check the SNR
commandwindow
disp(['The target SNR is ' num2str(SNR) ' dB.' newline ...
      'The actual SNR is ' num2str(snr(x, n)) ' dB.'])

% Plot all the noises on the same plot

p = sigma*pinknoise(N);
r = sigma*rednoise(N);
b = sigma*bluenoise(N);
v = sigma*violetnoise(N);

figure(2)
hold on;
plot(t, p, 'r', 'LineWidth', 1.5)
grid on
plot(t, r, '--b', 'LineWidth', 1.0)
plot(t, b, '--g', 'LineWidth', 1.0)
plot(t, v, '--m', 'LineWidth', 1.0)
xlim([0 max(t)])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Amplitude')
title('Signal + Noise in the time domain')
legend('Pink Noise','Red Noise','Blue Noise','Violet Noise')

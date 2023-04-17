clear, clc, close all

% signal parameters
fs = 44100;         % sampling frequency, Hz
T = 10;             % signal duration, s
N = round(fs*T);    % number of samples

% noise generation
x = rednoise(N);    % red noise, PSD falls off by -20 dB/dec

% calculate the noise PSD
winlen = 2*fs;
window = hanning(winlen, 'periodic');
noverlap = winlen/2;
nfft = winlen;

[Pxx, f] = pwelch(x, window, noverlap, nfft, fs, 'onesided');
PxxdB = 10*log10(Pxx);

% plot the noise PSD
figure(1)
semilogx(f, PxxdB, 'r', 'LineWidth', 1.5)
grid on
xlim([1 max(f)])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Frequency, Hz')
ylabel('Magnitude, dBV^{2}/Hz')
title('Power Spectral Density of the Noise Signal')

% Repeat the same for all the noises

w = wgn(N,1,0);
p = pinknoise(N);
r = rednoise(N);
b = bluenoise(N);
v = violetnoise(N);

% calculate the noise PSD for all the noises
winlen = 2*fs;
window = hanning(winlen, 'periodic');
noverlap = winlen/2;
nfft = winlen;

[Pxx_w, f] = pwelch(w, window, noverlap, nfft, fs, 'onesided');
PxxdB_w = 10*log10(Pxx_w);

[Pxx_p, f] = pwelch(p, window, noverlap, nfft, fs, 'onesided');
PxxdB_p = 10*log10(Pxx_p);

[Pxx_r, f] = pwelch(r, window, noverlap, nfft, fs, 'onesided');
PxxdB_r = 10*log10(Pxx_r);

[Pxx_b, f] = pwelch(b, window, noverlap, nfft, fs, 'onesided');
PxxdB_b = 10*log10(Pxx_b);

[Pxx_v, f] = pwelch(v, window, noverlap, nfft, fs, 'onesided');
PxxdB_v = 10*log10(Pxx_v);

% plot the noise PSD for all the noises
figure(2)
semilogx(f, PxxdB_p, 'r', 'LineWidth', 1.5)
hold on
semilogx(f, PxxdB_w, 'cyan', 'LineWidth', 1.5)
semilogx(f, PxxdB_r, 'b', 'LineWidth', 1.5)
semilogx(f, PxxdB_b, 'g', 'LineWidth', 1.5)
semilogx(f, PxxdB_v, 'm', 'LineWidth', 1.5)
grid on
xlim([1 max(f)])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Frequency, Hz')
ylabel('Magnitude, dBV^{2}/Hz')
title('Power Spectral Density of the Noise Signal')
legend('Pink Noise','White_Noise','Red Noise','Blue Noise','Violet Noise')




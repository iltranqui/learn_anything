using Plots
using FFTViews
using FFTW

dt = 0.01 # Time step
t = 0:dt:2 # Time vector

fmin = 0.1 # Minimum frequency
fmax = 2.0 # Maximum frequency

function chirp_signal(t, fmin, fmax)
    return sin.(2π .* (fmin .+ (fmax - fmin) .* t))
end

x = chirp_signal(t, fmin, fmax)

X = fft(x)

f = (0:length(x) - 1) / (dt * length(x))

x_rec = ifft(X)

plot(t, x, label="Original signal", color=:blue)
plot!(t, x_rec, label="Reconstructed signal", linestyle=:dash, color=:red)
title!("Comparison of original signal and reconstructed signal")
xlabel!("Time (s)")
ylabel!("Amplitude")
legend!()

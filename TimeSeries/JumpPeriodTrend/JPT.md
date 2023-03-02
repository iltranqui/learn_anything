# JUMP-TREND-PERIOD

To check for Jumps the 

# Jump - Mann–Whitney test 

The nonparametric MW test is another reliable test used to investigate numerically abrupt alternations in time series. The test is performed between 2 different sample of the timeseries, here called sample1 and sample2

$ U_{M W}=\frac{\sum_{t=1}^{N_1}\left(R(g(t))-\frac{n_1\left(n_1+n_2+1\right)}{2}\right)}{\sqrt{\frac{n_1 n_2\left(n_1+n_2+1\right)}{12}}} $

A corresponding probability to the test statistic $(P_{|UMW |})$
greater than 5% means the existence of a significant Jump in the time series. The MW
test is also termed MW–Wilcoxon test or the Wilcoxon Rank–Sum test


```matlab
%matlab tests
mwSTATS= mwwtest(sample1,sample2);
```
sample1 and sample2 are row vectors so that “sample1 + sample 2 = input”. If
the number of combinations is less than 20,000, the algorithm calculates the exact
ranks distribution; otherwise, it will use a normal distribution approximation. 

The ranksum function is a built-in MATLAB function that is used to apply the
MW test to time series

```matlab
[pVal H stat]=ranksum(sample1,sample2, 'argument' , value)
 if mwSTATS.p(1,2) < 0.01
 disp(' significant JUMP is detected in the series')
 else
 disp(' significant JUMP is NOT detected in the series')
end
```

To apply these to a time series here there is an [example](TimeSeries/JumpPeriodTren/Jump_TimeSeries.m)

# Period - Fisher's g Test

One method used for the detection of periodicity in a time series is to find significant
peaks in periodogram (Iω) of the time series

The periodogram is calculated as follows:

$ 
I_\omega=\frac{1}{N}\left|\sum_{t=1}^N x(t) \exp (-i \omega t)\right|^2 \quad \omega=[0-\pi] $ 

a significant peak  is expected to occur in $\omega=2 \pi k / n, k=0,1,2, \ldots, n / 2 $. The statistic for maximum periodogram denoting the significant period is calculated as: 

$ g^*=\frac{\max _{\cdot k}\left(I_{\omega k}\right)}{\sum_{k=1}^{n / 2} I_{\omega k}}  $ where $ g^*$ is a Fisher's g statistic. 

Now lot's of mathematics ... Bla bla bla 

The periodicity related to $ \Omega_{z} $ is
significant if the critical value of F distribution at a significant level (F(2, N−2))is lower
than $ F^* $: $ F^∗ ≥ F(2, N − 2) $

* if pval < 0.05 -> Periodicity is not detected
* if pval > 0.05 -> Periodicity is detected

To apply these to a time series here there is an [example](TimeSeries/JumpPeriodTren/Period_TimeSeries.m)

The periodogram (Introduced before R2006a) function returns the power spectral
density estimate of the time series (Pxx) and the frequencies (F). Then the maximum
periodogram is found to obtain the g-statistic. Finally, the P-value of the statistic is
calculated based on Equation above.

# Trend - Mann-Kendall Test

*H0 = Trend absence
*H1 = Trend Presence


```matlab
[H,p_value, S, StdS]=Mann_Kendall(input,alpha)
```

* H = 1 ->  the null hypothesis is rejected -> Trend Present
* H = 0 -> indicates a failure to reject the null hypothesis -> Trend Absent/Not Present

Link to the file to include in the working directory [Mann-Kendall Test](TimeSeries/JumpPeriodTren/Mann_Kendall.m)


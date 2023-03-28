# Stochastic Modeling

There are 3 different kinds of Stochastic Modeling

* Determinstic 
* Probabilistic
* stochastic Modeling
 
# Deterministic models

* models do not involve any random or probabilistic changes in their behavior.

Deterministic models are statistical models whose results are obtained based on a series of known relationships between states and
phenomena. Deterministic models use a set of rules which yield
the same outcome given the same set of initial conditions/inputs. This is because these
models do not involve any random or probabilistic changes in their behavior.
These models provide a single possible outcome for the phenomena, such as the amount of
rainfall on a given day in a basin. Deterministic models can use both linear and nonlinear
relationships to model and analyze the relationships between variables

# Probabilistic statistical models

* the data are independent of each other and time.

Probabilistic models, in contrast to deterministic models, consider the fact that randomness may play a role in determining future outcomes. From this, probabilistic
models combine random variables and probabilistic distributions to model phenomena,
expressing the randomness in series using distributions, such as normal distribution,
normal log, gamma, and Pearson. These models assume that the data are independent
of each other and time. For example, using probabilistic models, the probability of a
flood with a return period of 100 years could be predicted. Note that in this example, a
flood event is likely to be independent of an earlier flood at a different time.

# Stochastic concepts 

When stochastic models are compared to deterministic models, one considerable difference is that the same set of input will not strictly yield the same outputs. This is
due to the probabilistic considerations

The conventional stochastic statistical models are: 

* auto-regressive moving average (ARMA), 
* auto-regressive integrated moving average (ARIMA), 
* seasonal autoregressive integrated moving average (SARIMA),

>  The primary assumption in the development of stochastic models is the stationarity of the time series. However, the majority of time series that we collect from real-life
application do not follow this assumption

For this reason, we define a differencing operator to stationarize time series with deterministic terms. 
$ \begin{gathered}\nablax_t=x_t=x_t-x_{t-1}\end{gathered} $ sometimes with a delay time operator $ \begin{gathered}\nabla^1 x_t=(1-B) x_t \\=x_t-x_{t-1}\end{gathered} $ this is the *NonSeasonal Differencing* 
The *Seasonal Differencing* is.. 

### ARMA MOdels

The ARMA model is suitable for modeling nonseasonal time series with no differencing due to the use of nonseasonal parameters. The series should be stationarized with
a suitable preprocessing method. The differencing operator in ARIMA model has already been made suitable for application with nonstationary series. If seasonal differencing is applied instead, the ARIMA is also suitable to be used for seasonal time series. When the ARIMA model is modified to consider seasonal differencing it is known as the SARIMA model, which includes both seasonal and nonseasonal parameters and differencing operators

# Identify appropriate models and parameters’ orders

Use s ACF and PACF plots to dtermine if a univaraite time series is stationary. If not, preprocessing is necessary. Example: The YJ transform made the distribution of the time series more similar to normal distribution, which covers the condition of having normality of the data for stochastic modeling.


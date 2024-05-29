## Pay attention is 4gb 

https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt

# Generative Model 

A Generative model learns a probability distribution of the data set so that we can sample from the distribution to create new instances of data. Ex: if we have images of cats, we train a generative model on the images so that we can generate new images of cats.

The Entirety of all the pictures within an image is a huge collection of joint probability distributions. Each single pixel is a joint probability distribution of the pixel and all the other pixels in the image.
If you remember from proability theoery, if you have 2 distruvtion x and y $P(x,y)$ is the joint probability distribution of x and y. If you have a joint probability distribution of x and y, you can get the marginal probability distribution of x by summing over all the values of y. $P(x) = \sum_y P(x,y)$

## Variational Autoencoder

A variational autoencoder is a generative model which generates a new image by adding noise and then trying to rebuild the original image by removing the noise, tryin to get close to the original image. There are 2 processes: ù

* Reverse Process: $p$ Neural Network -> removing noise from the image, trying to replicate the original image
* Forward Process: $q$ Fixed -> simply adding gaussian noise to the image

### Forward Process

a vector of betas $\beta$ is sampled from a normal distribution. Each $\beta$ represents the mean and the variance of the noise added at each step. The Forward Process will try to predict the mean and variance of the noise added to the image, in a reverse process. 
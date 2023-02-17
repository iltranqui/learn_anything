import tensorflow as tf
import numpy as np

# Define the PINN model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        # Define the layers of the neural network
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        # Define the input layer
        x, v, t = inputs
        # Concatenate the input variables
        u = tf.concat([x, v, t], axis=1)
        # Pass the input through the neural network layers
        u = self.dense1(u)
        u = self.dense2(u)
        u = self.dense3(u)
        # Define the output layer
        u = self.dense4(u)
        # Return the model predictions
        return u
    
# The PINN class defines a neural network with 3 hidden layers, # each with 20 neurons and hyperbolic tangent activation functions. 
# The input to the network consists of the position x, velocity v, and time t of the mass, spring, and damper system, 
# concatenated together. The output of the network is a single number representing the predicted displacement of the mass. 
# The call method of the PINN class defines the forward pass of the network, 
# where the input is passed through the layers of the network to produce the output.


# Define the loss function
def pinn_loss(model, x_data, v_data, t_data, f_data, m, b, k):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_data)
        tape.watch(v_data)
        tape.watch(t_data)
        inputs = [x_data, v_data, t_data]
        predictions = model(inputs)
        du_dx = tape.gradient(predictions, x_data)
        du_dv = tape.gradient(predictions, v_data)
        du_dt = tape.gradient(predictions, t_data)
    del tape
    f = m * du_dt + b * du_dv + k * du_dx - f_data
    mse = tf.reduce_mean(tf.square(predictions - tf.zeros_like(predictions)))
    mse_f = tf.reduce_mean(tf.square(f))
    loss = mse + lambda_ * mse_f
    return loss

# Define the training loop
def train(model, x_data, v_data, t_data, f_data, m, b, k, optimizer, num_epochs=1000, print_interval=100):
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            loss = pinn_loss(model, x_data, v_data, t_data, f_data, m, b, k)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if (epoch+1) % print_interval == 0:
            print("Epoch {} Loss {:.6f}".format(epoch+1, loss.numpy()))

# Define the problem parameters and training data
    m = 1.0
    b = 0.1
    k = 1.0
    lambda_ = 1.0
    num_points = 1000
    t = np.linspace(0, 10, num_points)
    x_data = np.random.rand(num_points, 1)
    v_data = np.random.rand(num_points, 1)
    t_data = np.reshape(t, (-1, 1))
    f_data = -b * v_data - k * x_data

# Create and train the PINN model
    model = PINN()
    optimizer = tf.keras.optimizers.Adam()
    train(model, x_data, v_data, t_data, f_data, m, b, k, optimizer, num_epochs=5000, print_interval=500)

# Evaluate the PINN model on a test dataset
    x_test = np.linspace(0, 1, 100)
    v_test = np.linspace(-1, 1, 100)
    t_test = np.linspace(0, 10, 100)
    xx, vv, tt = np.meshgrid(x_test, v_test, t_test)
                            
#Reshape test data
    x_test = np.reshape(xx, (-1, 1))
    v_test = np.reshape(vv, (-1, 1))
    t_test = np.reshape(tt, (-1, 1))

#Evaluate predictions
    inputs_test = [x_test, v_test, t_test]
    predictions_test = model(inputs_test)

#Reshape predictions
    predictions_test = np.reshape(predictions_test, (100, 100, 100))

#Plot the results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('t')
    ax.scatter(xx, vv, tt, c=predictions_test.flatten())
    plt.show()
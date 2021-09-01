#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Numpy & PyTorch
import numpy as np
import torch


# A tensor is a number, vector, matrix or any n-dimensional array.

# ## Problem Statement

# We'll create a model that predicts crop yeilds for apples (*target variable*) by looking at the average temperature, rainfall and humidity (*input variables or features*) in different regions. 
# 
# Here's the training data:
# 
# >Temp | Rain | Humidity | Prediction
# >--- | --- | --- | ---
# > 73 | 67 | 43 | 56
# > 91 | 88 | 64 | 81
# > 87 | 134 | 58 | 119
# > 102 | 43 | 37 | 22
# > 69 | 96 | 70 | 103
# 
# In a **linear regression** model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias :
# 
# ```
# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# ```
# 
# It means that the yield of apples is a linear or planar function of the temperature, rainfall & humidity.
# 
# 
# 
# **Our objective**: Find a suitable set of *weights* and *biases* using the training data, to make accurate predictions.

# ## Training Data
# The training data can be represented using 2 matrices (inputs and targets), each with one row per observation and one column for variable.

# In[4]:


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')


# In[5]:


# Target (apples)
targets = np.array([[56], 
                    [81], 
                    [119], 
                    [22], 
                    [103]], dtype='float32')


# Before we build a model, we need to convert inputs and targets to PyTorch tensors.

# In[4]:


# Convert inputs and targets to tensors


# ## Linear Regression Model (from scratch)
# 
# The *weights* and *biases* can also be represented as matrices, initialized with random values. The first row of `w` and the first element of `b` are use to predict the first target variable i.e. yield for apples, and similarly the second for oranges.

# In[5]:


# Weights and biases


# The *model* is simply a function that performs a matrix multiplication of the input `x` and the weights `w` (transposed) and adds the bias `b` (replicated for each observation).
# 
# $$
# \hspace{2.5cm} X \hspace{1.1cm} \times \hspace{1.2cm} W^T \hspace{1.2cm}  + \hspace{1cm} b \hspace{2cm}
# $$
# 
# $$
# \left[ \begin{array}{cc}
# 73 & 67 & 43 \\
# 91 & 88 & 64 \\
# \vdots & \vdots & \vdots \\
# 69 & 96 & 70
# \end{array} \right]
# %
# \times
# %
# \left[ \begin{array}{cc}
# w_{11} & w_{21} \\
# w_{12} & w_{22} \\
# w_{13} & w_{23}
# \end{array} \right]
# %
# +
# %
# \left[ \begin{array}{cc}
# b_{1} & b_{2} \\
# b_{1} & b_{2} \\
# \vdots & \vdots \\
# b_{1} & b_{2} \\
# \end{array} \right]
# $$

# In[7]:


# Define the model
mu = np.mean(inputs, 0)
sigma = np.std(inputs, 0)
#normalizing the input
inputs = (inputs-mu) / sigma
inputs = np.hstack((np.ones((targets.size,1)),inputs))
print(inputs.shape)


# The matrix obtained by passing the input data to the model is a set of predictions for the target variables.

# In[8]:


rg = np.random.default_rng(12)
w = rg.random((1, 4))
print(w)


# In[8]:


# Compare with targets


# Because we've started with random weights and biases, the model does not perform a good job of predicting the target varaibles.

# ## Loss Function
# 
# We can compare the predictions with the actual targets, using the following method: 
# * Calculate the difference between the two matrices (`preds` and `targets`).
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# 
# The result is a single number, known as the **mean squared error** (MSE).

# In[9]:


# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return np.sum(diff * diff) / diff.size


# In[13]:


# Compute loss
preds = model(inputs,w)
cost_initial = mse(preds, targets)
print("Cost before regression: ",cost_initial)


# In[12]:


def model(x,w):
    return x @ w.T


# The resulting number is called the **loss**, because it indicates how bad the model is at predicting the target variables. Lower the loss, better the model. 

# ## Compute Gradients
# 
# With PyTorch, we can automatically compute the gradient or derivative of the `loss` w.r.t. to the weights and biases, because they have `requires_grad` set to `True`.
# 
# More on autograd:  https://pytorch.org/docs/stable/autograd.html#module-torch.autograd

# In[17]:


# Compute gradients
def gradient_descent(X, y, w, learning_rate, n_iters):
    J_history = np.zeros((n_iters,1))
    for i in range(n_iters):
        h = model(X,w)
        diff = h - y
        delta = (learning_rate/targets.size)*(X.T@diff)
        new_w = w - delta.T
        w=new_w
        J_history[i] = mse(h, y)
    return (J_history, w)


# The gradients are stored in the `.grad` property of the respective tensors.

# In[12]:


# Gradients for weights


# In[13]:


# Gradients for bias


# A key insight from calculus is that the gradient indicates the rate of change of the loss, or the slope of the loss function w.r.t. the weights and biases. 
# 
# * If a gradient element is **postive**, 
#     * **increasing** the element's value slightly will **increase** the loss.
#     * **decreasing** the element's value slightly will **decrease** the loss.
# 
# 
# 
# 
# * If a gradient element is **negative**,
#     * **increasing** the element's value slightly will **decrease** the loss.
#     * **decreasing** the element's value slightly will **increase** the loss.
#     
# 
# 
# The increase or decrease is proportional to the value of the gradient.

# Finally, we'll reset the gradients to zero before moving forward, because PyTorch accumulates gradients.

# In[13]:





# ## Adjust weights and biases using gradient descent
# 
# We'll reduce the loss and improve our model using the gradient descent algorithm, which has the following steps:
# 
# 1. Generate predictions
# 2. Calculate the loss
# 3. Compute gradients w.r.t the weights and biases
# 4. Adjust the weights by subtracting a small quantity proportional to the gradient
# 5. Reset the gradients to zero

# In[14]:


# Generate predictions


# In[15]:


# Calculate the loss


# In[16]:


# Compute gradients


# In[17]:


# Adjust weights & reset gradients


# In[18]:


#print(w)


# With the new weights and biases, the model should have a lower loss.

# In[19]:


# Calculate loss


# ## Train for multiple epochs
# 
# To reduce the loss further, we repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an epoch.

# In[23]:


import matplotlib.pyplot as plt
n_iters = 500
learning_rate = 0.01

initial_cost = mse(model(inputs,w),targets)

print("Initial cost is: ", initial_cost, "\n")

(J_history, optimal_params) = gradient_descent(inputs, targets, w, learning_rate, n_iters)

print("Optimal parameters are: \n", optimal_params, "\n")

print("Final cost is: ", J_history[-1])


# In[19]:


plt.plot(range(len(J_history)), J_history, 'r')

plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()


# In[21]:


# Calculate error
preds = model(inputs,optimal_params)
cost_final = mse(preds, targets)
# Print predictions
print("Prediction:\n",preds)
# Comparing predicted with targets
print("Targets:\n",targets)


# In[22]:


# Print targets
print("Cost after linear regression: ",cost_final)
print("Cost reduction percentage : {} %".format(((cost_initial- cost_final)/cost_initial)*100))


# In[ ]:





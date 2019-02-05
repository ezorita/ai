#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
#import droso_loader
import pickle

####
## Layer functions
####

def linear(x):
   return x

def tanh(x):
   return np.tanh(x)

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

# Derivatives
def d_linear(x):
   return np.ones(x.shape)

def d_tanh(tanh_x):
   return 1-np.square(tanh_x)

def d_sigmoid(sigm_x):
   return sigm_x*(1-sigm_x)

deriv = {
   linear:  d_linear,
   tanh:    d_tanh,
   sigmoid: d_sigmoid
}

####
## Cost functions
####

def cross_entropy(x,z):
   return -np.sum(x*np.log(z) + (1-x)*np.log(1-z))

def mse(x,z):
   return np.mean(np.square(x-z))

# Derivatives
def d_cross_entropy(x,z):
   return (1-x)/(1-z)-x/z

def d_mse(x,z):
   return -2.0/len(x)*(x-z)

L_deriv = {
   cross_entropy: d_cross_entropy,
   mse: d_mse
}


class DenoisingAutoEncoder:

   '''
   Class Deep Network
   Constructor arguments:
     nu  (1D numpy array) the number of units in each layer (input layer first)
   '''
   def __init__(self, units, loss, funcs=None):
      # Network description
      self.nl = len(units)
      self.nu = units

      # Funcs
      self.loss_func = loss
      if funcs is None:
         self.func = [sigmoid]*self.nl
      else:
         if len(funcs) < self.nl - 1:
            raise ValueError('funcs must have dimension len(nu)-1')
         else:
            self.func = funcs

      # Weights
      self.b  = [np.zeros((i)) for i in units]
      self.W  = [(np.random.random((i,j))-0.5)/np.sqrt(i*j) for i,j in zip(units[:-1], units[1:])]

      # Weight gradients
      self.db = [np.zeros((i)) for i in units]
      self.dW = [np.zeros((i,j)) for i,j in zip(units[:-1], units[1:])]

   ####
   ## Class helper methods
   ####

   def __setRandomWeights(self):
      self.W = [(np.random.random((i,j))-0.5)/np.sqrt(i*j) for i,j in zip(self.nu[:-1], self.nu[1:])]

   def __setZeroWeights(self):
      self.W = [np.zeros((i,j)) for i,j in zip(self.nu[:-1], self.nu[1:])]

   def __setRandomBiases(self):
      self.b = [(np.random.random((i))-0.5)/np.sqrt(i) for i in self.nu]

   def __setZeroBiases(self):
      self.b  = [np.zeros((i)) for i in self.nu]

   def __resetGradients(self):
      for db in self.db:
         db.fill(0)
      for dW in self.dW:
         dW.fill(0)

   ####
   ## Autoencoder Noise
   ####

   def __apply_noise(self, d, noise, noise_p):
      if noise == 'randmask':
         d *= np.array(np.random.random(d.shape) > noise_p, dtype=int)
      elif noise == 'salt-pepper':
         idx = np.array(np.random.random(d.shape) < noise_p, dtype=bool)
         spv = np.array(np.random.random(np.sum(idx)) < 0.5, dtype=int)
         d[idx] = spv
      elif noise == 'randomize':
         idx = np.array(np.random.random(d.shape) < noise_p, dtype=bool)
         spv = np.random.rand(np.sum(idx))
         d[idx] = spv

      return d

         
   ####
   ## Loss Function
   ####

   def loss(self, data, l_top=None):
      if l_top is None:
         l_top = self.nl - 1
      if self.loss_func == mse:
         return self.__mse_loss(data, l_top)
      elif self.loss_func == cross_entropy:
         return self.__crossEntropy_loss(data, l_top)
      
   def __crossEntropy_loss(self, data, l_top):
      z = data.copy()
      for l in np.arange(l_top):
         z = self.func[l+1](np.dot(z, self.W[l]) + self.b[l+1])
      for l in np.arange(l_top-1,0,-1):
         z = self.func[l](np.dot(z, self.W[l].T) + self.b[l])

      # Simplified loss to avoid NaN
      zp = np.dot(z, self.W[0].T) + self.b[0]
      return np.mean(-np.sum(data*zp-np.log(1+np.exp(-zp))-zp, axis=1))

   def __mse_loss(self, data, l_top):
      z = data.copy()
      for l in np.arange(l_top):
         z = self.func[l+1](np.dot(z, self.W[l]) + self.b[l+1])
      for l in np.arange(l_top-1,-1,-1):
         z = self.func[l](np.dot(z, self.W[l].T) + self.b[l])

      return np.mean(np.square(data-z))

   ####
   ## BackPropagation
   ####

   def __updateGradients_AE_BP(self, d, l_beg, l_end):
      nW = (l_end-l_beg)*2
      # Initialize units
      W = [0.0]*nW
      dW = [0.0]*nW
      b = [0.0]*(nW+1)
      db = [0.0]*(nW+1)
      y = [0.0]*(nW+1)
      func = [None]*(nW+1)

      y[0] = d

      # Extend encoder-decoder structure
      for i,l in enumerate(np.arange(l_beg, l_end)):
         W[i] = self.W[l]
         W[nW-1-i] = self.W[l].T

      for i,l in enumerate(np.arange(l_beg, l_end+1)):
         b[i] = self.b[l]
         b[nW-i] = self.b[l]
         func[i] = self.func[l]
         func[nW-i] = self.func[l]

      # Status of Units
      for i in np.arange(nW):
         y[i+1] = func[i+1](np.dot(y[i], W[i]) + b[i+1])

      ## Gradients
      # Top delta
      delta = L_deriv[self.loss_func](y[0], y[-1]) * deriv[func[-1]](y[-1])
      # Propagate delta
      for l in np.arange(nW, 0, -1):
         db[l] = delta
         dW[l-1] = np.outer(y[l-1], delta)
         delta = np.dot(W[l-1], delta) * deriv[func[l-1]](y[l-1])

      # Combine gradients
      for i,l in enumerate(np.arange(l_beg, l_end)):
         self.dW[l] += dW[i] + dW[nW-1-i].T

      for i,l in enumerate(np.arange(l_beg, l_end+1)):
         self.db[l] += db[i] + (db[nW-i] if l < l_end else 0)
         

   # Training
   def train(self, data, valid, epochs, batchsize, LR, noise='salt-pepper', noise_p=0.3):
      '''
      Layerwise Stack+FineTune Strategy
      '''

      # Epochs
      if type(epochs) == int:
         epochs = [epochs]*self.nl
      elif len(epochs) != self.nl:
         raise ValueError('epochs must have dimension nl')
      
      # Set Random Weights and Zero Biases
      self.__setRandomWeights()
      self.__setZeroBiases()

      # Training strategy indices
      end = np.repeat(np.arange(1,self.nl), 2)[1:]
      beg = np.zeros(len(end), dtype=int)
      beg[np.arange(1,len(end),2)] = np.arange(1,self.nl-1)

      # Loss logs
      trL_log = []
      vlL_log = []

      # Stack + FineTune
      for l_beg, l_end in zip(beg,end):
         # Compute data at beg layer
         d_l = data.copy()
         for l in np.arange(l_beg):
            d_l = self.func[l+1](np.dot(d_l, self.W[l]) + self.b[l+1])
         
         for e in xrange(epochs[l_end]):
            # Shuffle data
            d_e = d_l.copy()
            np.random.shuffle(d_e)
            
            # Apply noise to training data
            d_e = self.__apply_noise(d_e, noise, noise_p)

            # Iterate
            for it, d in enumerate(d_e):
               # End of batch
               if it % batchsize == 0:
                  for l in np.arange(l_beg, l_end):
                     # Update weights and biases
                     self.W[l]   -= LR * self.dW[l] / float(batchsize)
                     self.b[l+1] -= LR * self.db[l+1] / float(batchsize)

                  # Reset gradient estimates
                  self.__resetGradients()
                  
                  # Verbose iteration
                  sys.stdout.write('\r[DAE] layers: {}-{} | epoch: {} | iteration: {}'.format(l_beg, l_end, e+1, it/batchsize+1))
                  sys.stdout.flush()

               # Compute gradients
               self.__updateGradients_AE_BP(d, l_beg, l_end)
               
            # Update weights and biases with last gradient
            remain = (len(d_e) % batchsize)
            if remain:
               for l in np.arange(l_beg, l_end):
                  self.W[l]   -= LR * self.dW[l] / float(remain)
                  self.b[l+1] -= LR * self.db[l+1] / float(remain)

            # End of epoch, compute loss
            trL = self.loss(data, l_end)
            vlL = self.loss(valid, l_end)
            trL_log.append(trL)
            vlL_log.append(vlL)
            
            # Verbose
            sys.stdout.write("\r[DAE] layers: {}-{} | epoch: {} | Training Loss: {:.3f} | Validation Loss: {:.3f}\n".format(l_beg, l_end, e+1, trL, vlL))

      return trL_log, vlL_log


def main():
   train, valid, test = mnist_loader.load_data()
   train_d, train_l = train
   valid_d, valid_l = valid
   test_d, test_l = test

   units = [28*28,200,200,200,3]
   funcs = [sigmoid, sigmoid, sigmoid, sigmoid, linear]

   dnet = DenoisingAutoEncoder(units=units, funcs=funcs, loss=mse)

   train_loss, valid_loss = dnet.train(data=train_d, valid=valid_d, epochs=10, batchsize=100, LR=0.01)

   # Plot Loss
   plt.figure()
   plt.plot(train_loss)
   plt.plot(valid_loss)
   plt.savefig('SDAE_training_loss.png', bbox_inches='tight', format='png',dpi=300)

   # Report Loss      
   sys.stdout.write("Loss:\n  Training:\t{:.3f}\n  Validation:\t{:.3f}\n  Test:\t{:.3f}\n".format(
      dnet.loss(train_d),
      dnet.loss(valid_d),
      dnet.loss(test_d)
   ))

   # Save trained network
   with open('sdae_mnist_784_200_200_200_3.dump', 'w') as f:
      pickle.dump(obj=dnet, file=f)

   # 3D scatter plot
   from mpl_toolkits.mplot3d import Axes3D
   from matplotlib import pyplot

   # Scatter plot of latent space (test data)
   z = test_d.copy()
   for l in np.arange(dnet.nl-1):
      z = dnet.func[l+1](np.dot(z, dnet.W[l]) + dnet.b[l+1])

   fig = pyplot.figure()
   ax = Axes3D(fig)
   ax.scatter(z[:,0],z[:,1],z[:,2], c=test_l, s=0.7)
   fig.savefig('sdae_latent_test.png', bbox_inches='tight', format='png',dpi=300)

   # Scatter plot of latent space (train data)
   z = train_d.copy()
   for l in np.arange(dnet.nl-1):
      z = dnet.func[l+1](np.dot(z, dnet.W[l]) + dnet.b[l+1])

   fig = pyplot.figure()
   ax = Axes3D(fig)
   ax.scatter(z[:,0],z[:,1],z[:,2], c=train_l, s=0.7)
   fig.savefig('sdae_latent_train.png', bbox_inches='tight', format='png',dpi=300)


if __name__ == '__main__':
   main()

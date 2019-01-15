#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
from sklearn.manifold import TSNE

def plot_digits(m, L=28, fname='digits'):
   samples = len(m)
   N = m[0].shape[0]
   pixels = np.zeros(((L+1)*samples+1,(L+1)*N+1))

   for s in range(samples):
      for n in range(N):
         pixels[(s*(L+1)+1):((s+1)*(L+1)),(n*(L+1)+1):((n+1)*(L+1))] = m[s][n,:].reshape((L,L))

   plt.imshow(1-pixels, cmap='gray', interpolation=None)
   plt.axis('off')
   plt.savefig('{}.png'.format(fname), bbox_inches='tight', format='png',dpi=300)


def sigmoid(x):
   return 1 / (1 + np.exp(x))

class DeepNet:
   '''
   Class Deep Network
   Constructor arguments:
     nu  (1D numpy array) the number of units in each layer (input layer first)
   '''
   def __init__(self, nu):
      # Network description
      self.nl = len(nu)
      self.nu = nu

      # Weights
      self.b  = [np.random.random((i)) for i in nu]
      self.W  = [np.random.random((i,j)) for i,j in zip(nu[:-1], nu[1:])]

      # Weight gradients
      self.b_grad = [np.zeros((i)) for i in nu]
      self.W_grad = [np.zeros((i,j)) for i,j in zip(nu[:-1], nu[1:])]

      # Loss function
      self.L = 0

   def __cprob(self, d, ld, lt):
      '''
      Computes the conditional probabilities in the l-th layer units
      given d at the input layer.
      Arguments:
        d   data sample
        ld  origin layer
        lt  target layer
      '''
      cp = d.copy()
      if ld < lt:
         # Forward
         for i in xrange(ld,lt,1):
            cp = sigmoid(np.dot(self.W[i].T, cp) + self.b[i+1])
      else:
         # Backward
         for i in xrange(ld,lt,-1):
            cp = sigmoid(np.dot(self.W[i-1], cp) + self.b[i-1])
            
      return cp

   def __resetGradients(self):
      for bg in self.b_grad:
         bg.fill(0)
      for Wg in self.W_grad:
         Wg.fill(0)
      self.L = 0

   def __updateGradients_CD(self, d, l, CDn=1):
      '''
      Computes the gradient of the l-th layer given the input data d, using Contrastive Divergence.
      Arguments:
        d    data sample (input layer)
        l    layer to be updated
        CDn  Contrastive Divergence order (default 1)
      '''
      ## Contrastive divergence
      # Initial samples
      sl0 = np.array(np.random.rand(self.nu[l]) < self.__cprob(d,0,l), dtype=int)
      sr0 = np.array(np.random.rand(self.nu[l+1]) < self.__cprob(sl0,l,l+1), dtype=int)
         
      # Draw samples from conditional probabilities (back and forth)
      sr = sr0
      for i in range(CDn):
         sl = np.array(np.random.rand(self.nu[l]) < self.__cprob(sr,l+1,l), dtype=int)
         sr = np.array(np.random.rand(self.nu[l+1]) < self.__cprob(sl,l,l+1), dtype=int)
           
      # Update weight gradients
      self.W_grad[l] += np.outer(sl,sr) - np.outer(sl0,sr0)
      
      # Update bias gradients
      self.b_grad[l+1] += sr - sr0
      if l == 0:
         self.b_grad[l] += sl - sl0

   def __updateGradients_DAE(self, d, l, noise='mask', noise_p=0.3, compute_L=True):

      # Compute sample at top layer
      x = d.copy()
      for i in xrange(l):
         x = sigmoid(np.dot(self.W[i].T, x) + self.b[i+1])

      # Add noise to training sample
      _x = x.copy()
      noise_sz = int(len(x)*noise_p)
      if noise == 'mask':
         _x[np.random.randint(0, len(x), size=noise_sz)] = 0
      elif noise == 'randomize':
         _x[np.random.randint(0, len(x), size=noise_sz)] = np.random.rand(noise_sz)

      # Parameters
      W = self.W[l].T
      by = self.b[l+1]
      bz = self.bz[l]

      # Compute values of network elements
      y = sigmoid(np.dot(W, _x) + by)
      z = sigmoid(np.dot(W.T, y) + bz)

      # Naive element-wise computation
      # W_grad = np.zeros(W.shape)
      # for i in range(W.shape[0]):
      #    for j in range(W.shape[1]):
      #       W_grad[i,j] = (z[j] * (1-x[j]) - x[j]*(1-z[j])) * y[i] * (1+W[i,j]*x[j]*(1-y[i]))
      # self.W_grad[l] += W_grad.T

      # by_grad = np.zeros(by.shape)
      # for i in range(W.shape[0]):
      #    for j in range(W.shape[1]):
      #       by_grad[i] += (z[j]*(1-x[j]) - x[j]*(1-z[j])) * W[i,j] * y[i] * (1-y[i])
      # self.b_grad[l+1] += by_grad

      # bz_grad = np.zeros(bz.shape)
      # for j in range(W.shape[1]):
      #    bz_grad[j] = z[j] * (1-x[j]) - x[j] * (1-z[j])
      # self.bz_grad[l] += bz_grad
      
      # Gradients
      xz = z*(1-x) - x*(1-z)
      self.W_grad[l]   += (np.outer(y, xz) * (1 + W * np.outer(1-y, x))).T
      self.b_grad[l+1] += y * (1-y) * np.dot(W, xz)
      self.bz_grad[l]  += xz

      # Loss Function
      if compute_L:
         #self.L += -np.sum(x*np.log(z) + (1-x)*np.log(1-z))
         self.L += np.sum((x-z)**2)
      
   def __preTrainRBM(self, data, epochs, batchsize, LR=0.01, **algopts):
      '''
      Pretrain neural network using RBM algorithm (Contrastive Divergence)
      '''
      # Layer-wise greedy training
      for layer in xrange(self.nl-1):
         for e in xrange(epochs):
            # Reset Gradients
            self.__resetGradients()
            # Reshuffle input data
            np.random.shuffle(data)
            for it, d in enumerate(data):
               # End of batch
               if it % batchsize == 0:
                  # Update weights and biases
                  for b, bg in zip(self.b, self.b_grad):
                     b += LR * bg / float(batchsize)
                  for W, Wg in zip(self.W, self.W_grad):
                     W += LR * Wg / float(batchsize)
                  # Reset gradient estimates
                  self.__resetGradients()
                  # Verbose
                  sys.stdout.write('\r[pre-train] alg: DBN | layer: {} | epoch: {} | iteration: {}   '.format(layer+1, e+1, it/batchsize+1))
                  sys.stdout.flush()

               # Update gradient estimates
               self.__updateGradients_CD(d, layer, **algopts)

            # Update weights and biases with last gradient
            remain = (len(data) % batchsize)
            if remain:
               # Update weights and biases
               for b, bg in zip(self.b, self.b_grad):
                  b += LR * bg / float(remain)
               for W, Wg in zip(self.W, self.W_grad):
                  W += LR * Wg / float(remain)

      # Verbose
      sys.stdout.write('\n')


   def __preTrainSDAE(self, data, epochs, batchsize, LR=0.01, **algopts):
      '''
      Pretrain network as a stack of Denoising AutoEncoders (sDAE)
      '''
      # Temp variables
      self.bz = [np.random.random((i)) for i in self.nu]
      self.bz_grad = [np.zeros((i)) for i in self.nu]
      
      # Backpropagation
      for layer in xrange(self.nl-1):
         for e in xrange(epochs):
            # Reset Gradients
            self.__resetGradients()
            # Reshuffle input data
            np.random.shuffle(data)
            for it, d in enumerate(data):
               # End of batch
               if it % batchsize == 0:
                  # Update weights and biases
                  self.W[layer]   += LR * self.W_grad[layer] / float(batchsize)
                  self.bz[layer]  += LR * self.bz_grad[layer] / float(batchsize)
                  self.b[layer+1] += LR * self.b_grad[layer+1] / float(batchsize)

                  # Update loss function as mean loss
                  L = self.L / float(batchsize)
                  # Reset gradient estimates
                  self.__resetGradients()
                  # Verbose
                  sys.stdout.write('\r[pre-train] alg: DAE | layer: {} | epoch: {} | iteration: {} | Loss: {}    '.format(layer+1, e+1, it/batchsize+1, L))
                  sys.stdout.flush()

               # Update gradient estimates
               self.__updateGradients_DAE(d, layer, **algopts)

            # Update weights and biases with last gradient
            remain = (len(data) % batchsize)
            if remain:
               # Update weights and biases
               self.b[layer+1] += LR * self.b_grad[layer+1] / float(remain)
               self.W[layer]   += LR * self.W_grad[layer] / float(remain)

      # Verbose
      sys.stdout.write('\n')
      

   def __preTrainTwoStep(self, data, epochs, batchsize, LR=0.01, **algopts):
      # Algoritme de pretraining amb two steps.
      pass
      

   def preTrain(self, data, epochs, batchsize, algorithm='RBM', **algopts):
      '''
      Unsupervised weight pre-training.

      Common options:
        - LR  Learning rate for stochastic gradient update [0.005]

      Algorithms:
        - Restricted Boltzmann Machine ('RBM')
          Layer-wise greedy training.
          Algorithm options:
            - CDn Contrastive divergence order [1]

        - Denoising Auto Encoder ('DAE')
          Layer-wise training of denoising autoencoders.
          Algorithm options:
            - noise      Noise type ('mask', 'randomize') ['mask']
            - noise_p    % of samples affected by noise [0.3]
            - compute_L  Compute Loss function (True/False) [True]
      
      '''
      if algorithm == 'RBM':
         self.__preTrainRBM(data, epochs, batchsize, **algopts)
      elif algorithm == 'DAE':
         self.__preTrainSDAE(data, epochs, batchsize, **algopts)
      else:
         raise ValueError('unknown algorithm')
      

   def sampleRepresentation(self, v, N=1):

      ### ERROR WARNING
      # This way of generating samples from the conditional probability P(h | v) is wrong.
      # Must use variational inference to estimate P(h | v), sample h given v, then estimate P(v | h)
      # in the same way and sample v given h.
      ###
      
      '''
      Computes N sampled iterations of the visible neurons of the RBM.
      Parameters: 
        v: Initial value of the visible nodes (nv-dimensional array).
        N: Number of iterations (consecutive samples)
      Return value: a matrix containing the original v and one sample per row.
      '''
            
      samples = np.array(v).reshape(1,self.nu[0])
      for i in range(N):
         v = np.array(np.random.rand(self.nu[0]) < v, dtype=int)
         h = np.array(np.random.rand(self.nu[self.nl-1]) < self.__cprob(v, 0, self.nl-1), dtype=int)
         v = self.__cprob(h, self.nl-1, 0)
         samples = np.concatenate((samples, np.array(v).reshape((1,self.nu[0]))), axis=0)

      return samples

         
def main():
   # Load MNIST data
   train, valid, test = mnist_loader.load_data()
   images, digits = train

   # Create RBM
   layers = np.array([28*28,100,100,100])
   mnist_dbn = DeepNet(layers)

   # Pre-train DeepNet as DBN
   #   mnist_dbn.preTrain(images, epochs=10, batchsize=100, algorithm='DBN', CDn=1, LR=0.5)
   mnist_dbn.preTrain(images, epochs=10, batchsize=100, algorithm='DAE', LR=0.5, noise_p=0.1)
   
   # Sample probabilities
   sample_set = []
   for s in range(10):
      sample_set.append(mnist_dbn.sampleRepresentation(images[s,:], N=20))
   plot_digits(sample_set, fname='digit_probs')
   
   # Random data
   sample_set = []
   for s in range(10):
      sample_set.append(mnist_dbn.sampleRepresentation(np.random.rand((28*28)), N=20))
   plot_digits(sample_set, fname='random_noise')


if __name__ == '__main__':
   main()

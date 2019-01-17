#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader

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
   return 1 / (1 + np.exp(-x))

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
      self.b  = [np.zeros((i)) for i in nu]
      self.W  = [(np.random.random((i,j))-0.5)/np.sqrt(i*j) for i,j in zip(nu[:-1], nu[1:])]

      # Weight gradients
      self.db = [np.zeros((i)) for i in nu]
      self.dW = [np.zeros((i,j)) for i,j in zip(nu[:-1], nu[1:])]

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
      for db in self.db:
         db.fill(0)
      for dW in self.dW:
         dW.fill(0)
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
      self.dW[l] += np.outer(sl,sr) - np.outer(sl0,sr0)
      
      # Update bias gradients
      self.db[l+1] += sr - sr0
      if l == 0:
         self.db[l] += sl - sl0

   def __updateGradients_DAE(self, d, l, noise='mask', noise_p=0.3, compute_L=True):

      # Compute sample at top layer
      x = d.copy()
      for i in xrange(l):
         x = sigmoid(np.dot(self.W[i].T, x) + self.b[i+1])

      # Add noise to training sample
      _x = x.copy()
      noise_sz = int(len(x)*noise_p)
      if noise == 'mask':
         _x[np.random.choice(np.arange(len(x)), size=noise_sz)] = 0
      elif noise == 'randomize':
         _x[np.random.choice(np.arange(len(x)), size=noise_sz)] = np.random.rand(noise_sz)

      # Parameters
      W = self.W[l].T
      by = self.b[l+1]
      bz = self.b[l]

      # Compute values of network elements
      y = sigmoid(np.dot(W, _x) + by)
      z = sigmoid(np.dot(W.T, y) + bz)
      
      # Gradients
      xz = z*(1-x) - x*(1-z)
      self.dW[l]   += np.outer(x, y*(1-y) * np.dot(W, xz)) + np.outer(xz, y)
      self.db[l+1] += y * (1-y) * np.dot(W, xz)
      if l == 0:
         self.db[l]+= xz

      # Loss Function
      if compute_L:
         self.L += -np.sum(x*np.log(z) + (1-x)*np.log(1-z))

         
   def __cprob_h_EO(self, s):
      '''
      Computes the Conditional probabilities of the hidden units given the visible
      units in even-odd topology.
      The only difference from a normal RBM is that the hidden and visible layers
      are not fully connected.

      Arguments:
        s  Samples
      '''
      # Energies of the hidden units
      E = [np.zeros(n) for n in self.nu]      
      for l in xrange(len(self.nu)-1):
         if l % 2 == 0:
            E[l+1] += np.dot(self.W[l].T,s[l])
         else:
            E[l] += np.dot(self.W[l],s[l+1])

      # Conditional probabilities of the hidden units
      Ph = [np.zeros(n) for n in self.nu]
      for l in np.arange(1,len(self.nu),step=2):
         Ph[l] = sigmoid(E[l] + self.b[l])

      return Ph


   def __cprob_v_EO(self, s):
      '''
      Computes the Conditional probabilities of the visible units given the hidden
      units in even-odd topology.
      The only difference from a normal RBM is that the hidden and visible layers
      are not fully connected.
      '''
      # Energies of the visible units
      E = [np.zeros(n) for n in self.nu]      
      for l in xrange(len(self.nu)-1):
         if l % 2 == 0:
            E[l] += np.dot(self.W[l],s[l+1])
         else:
            E[l+1] += np.dot(self.W[l].T,s[l])

      # Conditional probabilities of the visible units
      Pv = [np.zeros(n) for n in self.nu]
      for l in np.arange(len(self.nu),step=2):
         Pv[l] = sigmoid(E[l] + self.b[l])

      return Pv
      


   def __updateGradients_EO(self, d_eo, CDn=1):
      '''
      This is a special case of Contrastive Divergence, only used in stage2 of DBM
      pretraining. This works on even-odd RBMs, in which the visible layer is the
      concatenation of the even layers and the hidden layer is the concatenation of
      the odd layers.
      The only difference wrt a normal RBM is that the visible and the hidden layers
      are not fully connected.

      Arguments:
        d    A list of data matrices containing the input samples.
        CDn  Contrastive divergence order
      '''

      ## Initial samples
      # Sample visible layer
      v0 = [np.array(np.random.rand(len(d)) < d, dtype=int) if d is not None else None for d in d_eo]
      # Sample hidden layer
      Ph = self.__cprob_h_EO(d_eo) # <--- Never contains None (has 0-arrays instead).
      h0 = [np.array(np.random.rand(len(p)) < p, dtype=int) if p is not None else None for p in Ph]
      

      ## Contrastive divergence
      h = h0
      for i in range(CDn):
         Pv = self.__cprob_v_EO(h)
         v  = [np.array(np.random.rand(len(p)) < p, dtype=int) if p is not None else None for p in Pv]
         Ph = self.__cprob_h_EO(v)
         h  = [np.array(np.random.rand(len(p)) < p, dtype=int) if p is not None else None for p in Ph]
         
      # Update weight gradients
      for l in range(len(self.nu)-1):
         if l % 2 == 0:
            self.dW[l] += np.outer(v[l], h[l+1]) - np.outer(v0[l], h0[l+1])
         else:
            self.dW[l] += np.outer(h[l], v[l+1]) - np.outer(h0[l], v0[l+1])
      
      # Update bias gradients
      for l in range(len(self.nu)):
         if l % 2 == 0:
            self.db[l] += v[l] - v0[l]
         else:
            self.db[l] += h[l] - h0[l]

            
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
               # Update gradient estimates
               self.__updateGradients_CD(d, layer, **algopts)

               # End of batch
               if it % batchsize == 0:
                  # Update weights and biases
                  for b, db in zip(self.b, self.db):
                     b += LR * db / float(batchsize)
                  for W, dW in zip(self.W, self.dW):
                     W += LR * dW / float(batchsize)
                  # Reset gradient estimates
                  self.__resetGradients()
                  # Verbose
                  sys.stdout.write('\r[pre-train] alg: DBN | layer: {} | epoch: {} | iteration: {}   '.format(layer+1, e+1, it/batchsize+1))
                  sys.stdout.flush()

            # Update weights and biases with last gradient
            remain = (len(data) % batchsize)
            if remain:
               # Update weights and biases
               for b, bg in zip(self.b, self.db):
                  b += LR * bg / float(remain)
               for W, Wg in zip(self.W, self.dW):
                  W += LR * Wg / float(remain)

      # Verbose
      sys.stdout.write('\n')


   def __preTrainSDAE(self, data, epochs, batchsize, LR=0.01, **algopts):
      '''
      Pretrain network as a stack of Denoising AutoEncoders (sDAE)
      '''
      # Temp variables
      self.bz = [np.random.random((i)) for i in self.nu]
      
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
                  self.W[layer]   -= LR * self.dW[layer] / float(batchsize)
                  self.b[layer]   -= LR * self.db[layer] / float(batchsize)
                  self.b[layer+1] -= LR * self.db[layer+1] / float(batchsize)

                  # Update loss function as mean loss
                  L = self.L / float(batchsize)
                  # Reset gradient estimates
                  self.__resetGradients()
                  # Verbose
                  sys.stdout.write('\r[pre-train] alg: DAE | layer: {} | epoch: {} | iteration: {} | Loss: {}       '.format(layer+1, e+1, it/batchsize+1, L))
                  sys.stdout.flush()

               # Update gradient estimates
               self.__updateGradients_DAE(d, layer, **algopts)

            # Update weights and biases with last gradient
            remain = (len(data) % batchsize)
            if remain:
               # Update weights and biases
               self.b[layer]   -= LR * self.db[layer] / float(remain)
               self.b[layer+1] -= LR * self.db[layer+1] / float(remain)
               self.W[layer]   -= LR * self.dW[layer] / float(remain)

      # Verbose
      sys.stdout.write('\n')
      

   def __preTrainDBM(self, data, epochs, batchsize, LR=0.01, s1_alg='DAE', s1_epochs=None, s1_batchsize=None, s1_LR=None, **algopts):
      '''
      Pretrain network with Kyunghyun Cho's two-stage algorithm
      '''

      # Arguments
      s1_epochs = epochs if s1_epochs is None else s1_epochs
      s1_batchsize = batchsize if s1_batchsize is None else s1_batchsize
      s1_LR = LR if s1_LR is None else s1_LR

      ## Stage 1
      # Select layers 0,2,4...
      even_idx = np.arange(len(self.nu),step=2)
      even_nu = self.nu[even_idx]
      stage1net = DeepNet(even_nu)
      # Pretrain with s1_alg
      stage1net.preTrain(data, s1_epochs, s1_batchsize, LR=s1_LR, algorithm=s1_alg)
      # Sample training data
      tData = [data]
      for l in range(len(even_nu)-1):
         tData.append(np.dot(tData[l], stage1net.W[l]))

      ## Stage 2
      # Train parameters with odd-even RBM (even layers are visible, odd hidden)
      data_idx = np.arange(len(data))
      d_eo = [None]*len(self.nu)
      for s1_l, l in enumerate(even_idx):
         d_eo[l] = tData[s1_l]

      # Train EO RBM
      for e in xrange(epochs):
         # Reset Gradients
         self.__resetGradients()
         # Reshuffle input data
         np.random.shuffle(data_idx)
         for l in even_idx:
            d_eo[l] = d_eo[l][data_idx,:]
         # Iterate over data
         for it in xrange(len(data)):
            # Update gradient estimates
            d = [sample[it,:] if sample is not None else None for sample in d_eo]

            self.__updateGradients_EO(d, **algopts)

            # End of batch
            if it % batchsize == 0:
               # Update weights and biases
               for b, db in zip(self.b, self.db):
                  b += LR * db / float(batchsize)
               for W, dW in zip(self.W, self.dW):
                  W += LR * dW / float(batchsize)
               # Reset gradient estimates
               self.__resetGradients()
               # Verbose
               sys.stdout.write('\r[pre-train] alg: EO-RBM | epoch: {} | iteration: {}   '.format(e+1, it/batchsize+1))
               sys.stdout.flush()

         # Update weights and biases with last gradient
         remain = (len(data) % batchsize)
         if remain:
            # Update weights and biases
            for b, bg in zip(self.b, self.db):
               b += LR * bg / float(remain)
            for W, Wg in zip(self.W, self.dW):
               W += LR * Wg / float(remain)

      # Verbose
      sys.stdout.write('\n')   

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
      elif algorithm == 'DBM':
         self.__preTrainDBM(data, epochs, batchsize, **algopts)
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
      
      W1 = self.W[0]
      W2 = self.W[1]
      W3 = self.W[2]
      b0 = self.b[0]
      b1 = self.b[1]
      b2 = self.b[2]
      b3 = self.b[3]

      samples = np.array(v).reshape(1,self.nu[0])
      for i in range(N):
         #v = np.array(np.random.rand(self.nu[0]) < v, dtype=int)
         #h = np.array(np.random.rand(self.nu[self.nl-1]) < self.__cprob(v, 0, self.nl-1), dtype=int)
         #v = self.__cprob(self.__cprob(v, 0, self.nl-1), self.nl-1, 0)
         mu1 = np.random.rand(len(b1))
         mu2 = np.random.rand(len(b2))
         mu3 = np.random.rand(len(b3))
         for j in range(20):
            mu1 = sigmoid(np.dot(W1.T, v) + np.dot(W2, mu2) + b1) 
            mu2 = sigmoid(np.dot(W2.T, mu1) + np.dot(W3, mu3) + b2) 
            mu3 = sigmoid(np.dot(W3.T, mu2) + b3) 
         h1 = np.random.rand(len(b1)) < mu1
         v = sigmoid(np.dot(W1, h1) + b0)
         samples = np.concatenate((samples, np.array(v).reshape((1,self.nu[0]))), axis=0)


      return samples

         
def main():
   # Load MNIST data
   train, valid, test = mnist_loader.load_data()
   images, digits = train

   # Create DeepNet
   layers = np.array([28*28,300,300,300])
   #layers = np.array([28*28,100,100])
   #layers = np.array([28*28,200])
   deepnet = DeepNet(layers)

   # Pre-train DeepNet as DBN
   deepnet.preTrain(images, epochs=2, batchsize=100, algorithm='DBM', CDn=1, LR=0.1)
   # deepnet.preTrain(images, epochs=3, batchsize=100, algorithm='DAE', LR=0.05, noise_p=0.5)
   
   # Sample probabilities
   sample_set = []
   for s in range(10):
      sample_set.append(deepnet.sampleRepresentation(images[s,:], N=20))
   plot_digits(sample_set, fname='digit_probs')
   
   # Random data
   sample_set = []
   for s in range(10):
      sample_set.append(deepnet.sampleRepresentation(np.random.rand((28*28)), N=20))
   plot_digits(sample_set, fname='random_noise')


if __name__ == '__main__':
   main()

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
      self.b  = [np.zeros((i)) for i in nu]
      self.W  = [np.zeros((i,j)) for i,j in zip(nu[:-1], nu[1:])]

      # Weight gradients
      self.b_grad = [np.zeros((i)) for i in nu]
      self.W_grad = [np.zeros((i,j)) for i,j in zip(nu[:-1], nu[1:])]

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

      
   def __preTrainDBN(self, data, epochs, batchsize, LR=0.01, **algopts):

      ### TODO
      # Allow the selection of the algorithm to compute P(v,h) (the negative part of the gradient)
      # Currently implemented is CD.
      # Could also implement MCMC methods.
      # Implement them in separate functions so that they can be used in other methods.
      # These methods only depend on the current parameters (and a current state, if not reset every iteration)
      ###
      
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

   def __preTrainDBM(self, data, epochs, batchsize, LR=0.01, **algopts):
      

   def preTrain(self, data, epochs, batchsize, algorithm='DBN', **algopts):
      '''
      Unsupervised weight pre-training.
      Algorithms:
        - Deep Belief Network ('DBN')
          Layer-wise greedy training.
          Algorithm options:
            - CDn Contrastive divergence order (default 1)
            - LR  Learning rate (default 0.005)
      '''
      if algorithm == 'DBN':
         self.__preTrainDBN(data, epochs, batchsize, **algopts)
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
   mnist_dbn.preTrain(images, epochs=10, batchsize=100, algorithm='DBN', CDn=1, LR=0.5)
   
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

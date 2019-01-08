#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader

def sigmoid(x):
   return 1 / (1 + np.exp(x))


def plot_digits(m, L=28, fname='digits'):
   samples = len(m)
   N = m[0].shape[0]
   pixels = np.zeros(((L+1)*samples+1,(L+1)*N+1))

   for s in range(samples):
      for n in range(N):
         pixels[(s*(L+1)+1):((s+1)*(L+1)),(n*(L+1)+1):((n+1)*(L+1))] = m[s][n,:].reshape((L,L))

   plt.imshow(1-pixels, cmap='gray', interpolation=None)
   plt.axis('off')
   plt.savefig('{}.png'.format(fname), bbox_inches='tight', format='png', dpi=L)

   
class RBM:
   def __init__(self,nv,nh,LR=0.005,CDn=1):
      self.nv = nv
      self.nh = nh
      self.W  = np.zeros((nv,nh))
      self.vb = np.zeros((nv))
      self.hb = np.zeros((nh))
      self.LR = LR
      self.CDn = CDn

   def v_cond_probs(self, h):
      return sigmoid(np.dot(self.W, h) + self.vb)

   def h_cond_probs(self, v):
      return sigmoid(np.dot(self.W.T, v) + self.hb)

   def update(self, v_sample):
      # Discretize input sample
      v0 = np.array(np.random.rand(self.nv) < v_sample, dtype=int)
      # Draw a sample using conditional probabilities P(h[i]=1 | v0)
      h0 = np.array(np.random.rand(self.nh) < self.h_cond_probs(v0), dtype=int)

      # Contrastive Divergence (order CDn)
      v = v0.copy()
      h = h0.copy()
      for i in range(self.CDn):
         # Draw samples from conditional probabilities (back and forth)
         v = np.array(np.random.rand(self.nv) < self.v_cond_probs(h))
         h = np.array(np.random.rand(self.nh) < self.h_cond_probs(v))

      # Update weights and biases
      self.W  += self.LR * (np.outer(v,h) - np.outer(v0,h0))
      self.vb += self.LR * (v - v0)
      self.hb += self.LR * (h - h0)

   def sample(self, v, N=1):
      '''
      Computes N sampled iterations of the visible neurons of the RBM.
      Parameters: 
        v: Initial value of the visible nodes (nv-dimensional array).
        N: Number of iterations (consecutive samples)
      Return value: a matrix containing the original v and one sample per row.
      '''
      samples = np.array(v).reshape(1,self.nv)
      for i in range(N):
         h = np.array(np.random.rand(self.nh) < self.h_cond_probs(v))
         v = np.array(np.random.rand(self.nv) < self.v_cond_probs(h))
         samples = np.concatenate((samples, np.array(v).reshape((1,self.nv))), axis=0)

      return samples

   def sample_probs(self, v, N=1):
      '''
      Computes N sampled iterations of the visible neurons of the RBM.
      Parameters: 
        v: Initial value of the visible nodes (nv-dimensional array).
        N: Number of iterations (consecutive samples)
      Return value: a matrix containing the original v and one sample per row.
      '''
      samples = np.array(v).reshape(1,self.nv)
      for i in range(N):
         h = np.array(np.random.rand(self.nh) < self.h_cond_probs(v))
         vp = self.v_cond_probs(h)
         v = np.array(np.random.rand(self.nv) < vp)
         samples = np.concatenate((samples, np.array(vp).reshape((1,self.nv))), axis=0)

      return samples


def main():
   # Load MNIST data
   train, valid, test = mnist_loader.load_data()
   images, digits = train

   # Create RBM
   mnist_rbm = RBM(28*28, 50)
   for it, image in enumerate(images):
      mnist_rbm.update(image)
      if it % 1000 == 0:
         print it

   # Sample numbers
   sample_set = []
   for s in range(10):
      sample_set.append(mnist_rbm.sample(images[s,:], N=20))

   plot_digits(sample_set, fname='digits')

   # Sample probabilities
   sample_set = []
   for s in range(10):
      sample_set.append(mnist_rbm.sample_probs(images[s,:], N=20))

   plot_digits(sample_set, fname='digit_probs')

   # Random data
   sample_set = []
   for s in range(10):
      sample_set.append(mnist_rbm.sample(np.random.rand((28*28)), N=500))

   plot_digits(sample_set, fname='random_noise')
   

if __name__ == '__main__':
   main()

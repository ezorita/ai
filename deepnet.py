#!/usr/bin/env python
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import pickle
import multiprocessing
import functools

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

def iterateParticles_MP(particles, dnet, iterations=5):
   for p in particles:
      for iter in xrange(iterations):
         # Flip flop
         Ph = dnet._DeepNet__flip(p)
         for l in range(1,len(Ph),2):
            p[l] = np.array(np.random.random(len(Ph[l])) < Ph[l], dtype=int)
         Pv = dnet._DeepNet__flop(p)
         for l in range(0,len(Pv),2):
            p[l] = np.array(np.random.random(len(Pv[l])) < Pv[l], dtype=int)

def variationalExpectation_MP(dnet, data, queue, max_iters=10000, tol=1e-2,temp=1.0):
   mu = [np.zeros(n) for n in dnet.nu]
   mu[0] = data.copy()
   prev_mu = [x.copy() for x in mu]

   for i in xrange(max_iters):
      for l in xrange(1,dnet.nl):
         mu[l] = sigmoid(1.0/temp*(np.dot(dnet.W[l-1].T, mu[l-1]) +
                                   dnet.b[l] +
                                   (np.dot(dnet.W[l],mu[l+1]) if l+1 < dnet.nl else 0)))
      if np.max([np.max(np.abs(x)) for x in prev_mu[1:]]) < tol:
         break

      prev_mu = [x.copy() for x in mu]

   # Push to queue
   queue.put(mu)

   
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
      self.L = 0


   ####
   ## DeepNet Interface Methods
   ####

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

   def fineTune(self, data, n_particles=100, flip_iters=5, LR=0.005):
      '''
      Shitty prototype GF style that will require much more work
      to fix than it took to write. You wanted agile programming?
      Well, you've got it!
      '''

      particles = [[np.random.random(nu) for nu in self.nu] for i in xrange(n_particles)]
      its = len(data)
      for it,v in enumerate(data):
         print it

         # Variational Expectation of training sample
         mu = self.__variationalExpectation(v)

         if it % 100 == 0:
            particles.pop()
            particles.insert(0,[np.random.random(nu) for nu in self.nu])

         # Parallel flip-flop particle iteration
         for p in particles:
            for i in range(flip_iters):
               # Flip flop
               Ph = self.__flip(p)
               for l in range(1,len(Ph),2):
                  p[l] = np.array(np.random.random(len(Ph[l])) < Ph[l], dtype=int)
               Pv = self.__flop(p)
               for l in range(0,len(Pv),2):
                  p[l] = np.array(np.random.random(len(Pv[l])) < Pv[l], dtype=int)
               
         # Update. (Decreasing Learning Rate)
         LRf = (its-it)*LR/its
         for l in range(self.nl-1):
            self.W[l] += LRf * (np.outer(mu[l],mu[l+1]) - np.mean([np.outer(p[l],p[l+1]) for p in particles],axis=0))
         for l in xrange(self.nl):
            self.b[l] += LRf * (mu[l] - np.mean([p[l] for p in particles],axis=0))

         if it % 1000 == 0:
            with open('dbm_new_finetune_it{}.dump'.format(it), 'w') as f:
               pickle.dump(self, file=f)


   def fineTune_MT(self, data, n_particles=100, flip_iters=10, LR=0.005, cpus=16):
      '''
      Shitty prototype GF style that will require much more work
      to fix than it took to write. You wanted agile programming?
      Well, you've got it!
      '''
      particles = [[np.random.random(nu) for nu in self.nu] for i in xrange(n_particles)]

      its = len(data)
      for it,v in enumerate(data):
         print it

         # Iteration
         procs = []
         
         # Variational Expectation of traing Sample
         mu_q = multiprocessing.Queue(1)
         p = multiprocessing.Process(target=variationalExpectation_MP, kwargs={'dnet': self, 'data': v, 'queue': mu_q})
         procs.append(p)
         p.start()

         # Parallel flip-flop particle iteration
         n_procs = int(np.ceil(n_particles/cpus))
         particle_groups = [particles[(cpus*i):(cpus*(i+1))] for i in xrange(0, len(particles), n_procs)]

         for p_group in particle_groups:
            p = multiprocessing.Process(target=iterateParticles_MP, kwargs={'dnet': self, 'particles': p_group, 'iterations': flip_iters})
            procs.append(p)
            p.start()

         # Wait processes
         for p in procs:
            p.join()
         mu = mu_q.get()

         # Update (decreasing learning rate)
         LRf = (its-it)*LR/its
         for l in range(self.nl-1):
            self.W[l] += LRf * (np.outer(mu[l],mu[l+1]) - np.mean([np.outer(p[l],p[l+1]) for p in particles],axis=0))
         for l in xrange(self.nl):
            self.b[l] += LRf * (mu[l] - np.mean([p[l] for p in particles],axis=0))
            
         if it % 1000 == 0:
            with open('dbm_new_finetune_MT_it{}.dump'.format(it), 'w') as f:
               pickle.dump(self, file=f)

               
   def fineTune_sDAE(self, data, epochs=1, batchsize=100, LR=0.005, **algopts):
      # For epoch
      for e in xrange(epochs):
         # Reset Gradients
         self.__resetGradients()
         # Reshuffle input data
         np.random.shuffle(data)
         for it, d in enumerate(data):
            # Update gradient estimates
            self.__updateGradients_sDAE_BP(d, **algopts)
            # End of batch
            if it % batchsize == 0:
               # Update weights and biases
               for b, db in zip(self.b, self.db):
                  b -= LR * db / float(batchsize)
               for W, dW in zip(self.W, self.dW):
                  W -= LR * dW / float(batchsize)
               # Update loss function as mean loss
               L = self.L / float(batchsize)
               # Reset gradient estimates
               self.__resetGradients()
               # Verbose
               sys.stdout.write('\r[fine-tune] alg: sDAE-BP | epoch: {} | iteration: {} | Loss: {}       '.format(e+1, it/batchsize+1, L))
               sys.stdout.flush()
         # Update weights and biases with last gradient
         remain = (len(data) % batchsize)
         if remain:
            # Update weights and biases
            for b, db in zip(self.b, self.db):
               b -= LR * db / float(remain)
            for W, dW in zip(self.W, self.dW):
               W -= LR * dW / float(remain)

      # Verbose
      sys.stdout.write('\n')
               
   def generate_RBM(self, v, N=1):
      samples = np.array(v).reshape(1,self.nu[0])
      for i in range(N):
         v = np.array(np.random.rand(self.nu[0]) < v, dtype=int)
         h = np.array(np.random.rand(self.nu[self.nl-1]) < self.__cprob(v, 0, self.nl-1), dtype=int)
         v = self.__cprob(self.__cprob(v, 0, self.nl-1), self.nl-1, 0)
         samples = np.concatenate((samples, np.array(v).reshape((1,self.nu[0]))), axis=0)

      return samples
      

   def generate_DBM(self, d, N=1, step=1, **vexpargs):
      '''
      Computes N sampled iterations of the visible neurons of the RBM.
      Parameters: 
        d: Initial value of the visible nodes (nv-dimensional array).
        N: Number of iterations (consecutive samples)
      Return value: a matrix containing the original v and one sample per row.
      '''
      v = d.copy()
      samples = np.array(v).reshape(1,self.nu[0])
      for i in range(N*step):
         mu = self.__variationalExpectation(v, **vexpargs)
         h1 = np.array(np.random.rand(self.nu[1]) < mu[1], dtype=int)
         v = sigmoid(np.dot(self.W[0], h1) + self.b[0])
         if i % step == 0:
            samples = np.concatenate((samples, np.array(v).reshape((1,self.nu[0]))), axis=0)

      return samples

   def totalEnergy(self, x):
      # x is the current state of the machine. Format: [x,h1,h2,h3,...]
      # E = -x.T*W*h - x.T*b_x - h.T*b_h
      E = 0
      for l in xrange(self.nl):
         E -= (np.dot(np.dot(x[l].T, self.W[l]), x[l+1]) if l < self.nl-1 else 0)
         E -= np.inner(x[l], self.b[l]) 

      return E

   
   ####
   ## Restricted Boltzmann Machine
   ####

   ## RBM: Conditional Probabilities
   
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

   ## RBM: Gradient Update

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
      self.dW[l] += np.outer(sl0,sr0) - np.outer(sl,sr)
      
      # Update bias gradients
      self.db[l+1] += sr0 - sr
      if l == 0:
         self.db[l] += sl0 - sl
         

   ## RBM: Training
   
   def __preTrainRBM(self, data, epochs, batchsize, LR=0.01, **algopts):
      '''
      Pretrain neural network using RBM algorithm (Contrastive Divergence)
      '''
      # Reset Weights and Biases
      self.__setZeroWeights()
      self.__setZeroBiases()
         
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


   ####
   ## Denoising AutoEncoder
   ####

   ## sDAE: Fine-tune Gradient Update

   def __updateGradients_sDAE_BP(self, d, noise='mask', noise_p=0.3, compute_L=True):

      # Initialize units
      W = [None]*(self.nl*2-2)
      dW = [None]*(self.nl*2-2)
      b = [None]*(self.nl*2-1)
      db = [None]*(self.nl*2-1)
      y = [None]*(self.nl*2-1)

      # Add noise to training sample
      _x = d.copy()
      noise_sz = int(len(_x)*noise_p)
      if noise == 'mask':
         _x[np.random.choice(np.arange(len(_x)), size=noise_sz)] = 0
      elif noise == 'randomize':
         _x[np.random.choice(np.arange(len(_x)), size=noise_sz)] = np.random.rand(noise_sz)
      
      y[0] = _x

      # Extend encoder-decoder structure
      for i in np.arange(self.nl-1):
         W[i] = self.W[i]
         W[self.nl*2-3-i] = self.W[i].T

      for i in np.arange(self.nl):
         b[i] = self.b[i]
         b[self.nl*2-2-i] = self.b[i]

      # Status of Units
      for i in np.arange(1,self.nl*2-1):
         y[i] = sigmoid(np.dot(y[i-1], W[i-1]) + b[i])

      # Gradients
      l = self.nl*2-2
      delta = y[-1]*(1-d)-d*(1-y[-1])
      db[l] = delta
      for l in np.arange(l, 0, -1):
         dW[l-1] = np.outer(y[l-1], delta)
         delta = np.dot(W[l-1], delta) * y[l-1] * (1-y[l-1])
         db[l-1] = delta

      # Combine gradients
      for i in np.arange(self.nl-1):
         self.dW[i] = dW[i] + dW[self.nl*2-3-i].T

      for i in np.arange(self.nl):
         self.db[i] = db[i] + db[self.nl*2-2-i]

      # Loss Function
      if compute_L:
         self.L += -np.sum(d*np.log(y[-1]) + (1-d)*np.log(1-y[-1]))

      
   
   def __updateGradients_sDAE(self, d, noise='mask', noise_p=0.3, compute_L=True):
      # Initialize units
      y = [None]*(self.nl*2-1)
      
      # Add noise to training sample
      _x = d.copy()
      noise_sz = int(len(_x)*noise_p)
      if noise == 'mask':
         _x[np.random.choice(np.arange(len(_x)), size=noise_sz)] = 0
      elif noise == 'randomize':
         _x[np.random.choice(np.arange(len(_x)), size=noise_sz)] = np.random.rand(noise_sz)
      
      y[0] = _x

      ## Unit status
      # Compute Encoder units
      l = 1
      for i in np.arange(self.nl-1):
         y[l] = sigmoid(np.dot(y[l-1],self.W[i]) + self.b[i+1])
         l += 1

      # Compute Decoder units
      for i in np.arange(self.nl-2,-1,step=-1):
         y[l] = sigmoid(np.dot(self.W[i],y[l-1]) + self.b[i])
         l += 1

      # Gradients (backpropagation)
      l = len(y)-1
      delta = y[-1]*(1-d) - d*(1-y[-1])
      
      # Backpropagate in Decoder
      for i in np.arange(self.nl-1):
         self.db[i] += delta
         self.dW[i] += np.outer(delta, y[l-1])
         delta = np.dot(delta, self.W[i]) * y[l-1] * (1-y[l-1])
         l -= 1

      # Backpropagate in Encoder
      for i in np.arange(self.nl-2,-1,step=-1):
         self.db[i+1] += delta
         self.dW[i] += np.outer(y[l-1], delta)
         delta = np.dot(self.W[i], delta) * y[l-1] * (1-y[l-1])
         l -= 1

      self.db[0] += delta

      # Loss Function
      if compute_L:
         self.L += -np.sum(d*np.log(y[-1]) + (1-d)*np.log(1-y[-1]))
      
   ## DAE: Gradient Update
   
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


   ## DAE: Training

   def __preTrainSDAE(self, data, epochs, batchsize, LR=0.01, **algopts):
      '''
      Pretrain network as a stack of Denoising AutoEncoders (sDAE)
      '''
      # Set Random Weights and Zero Biases
      self.__setRandomWeights()
      self.__setZeroBiases()
         
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



   ####
   ## Deep Boltzmann Machine
   ####

   ## DBM: EO-Conditional Probabilities
   
   def __flip(self, s, beta=1.0):
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
      for l in xrange(self.nl-1):
         if l % 2 == 0:
            E[l+1] += np.dot(self.W[l].T,s[l])
         else:
            E[l] += np.dot(self.W[l],s[l+1])

      # Conditional probabilities of the hidden units
      Ph = [np.zeros(n) for n in self.nu]
      for l in np.arange(1,self.nl,step=2):
         Ph[l] = sigmoid(beta*(E[l] + self.b[l]))

      return Ph


   def __flop(self, s, beta=1.0):
      '''
      Computes the Conditional probabilities of the visible units given the hidden
      units in even-odd topology.
      The only difference from a normal RBM is that the hidden and visible layers
      are not fully connected.
      '''
      # Energies of the visible units
      E = [np.zeros(n) for n in self.nu]      
      for l in xrange(self.nl-1):
         if l % 2 == 0:
            E[l]   += np.dot(self.W[l],s[l+1])
         else:
            E[l+1] += np.dot(self.W[l].T,s[l])

      # Conditional probabilities of the visible units
      Pv = [np.zeros(n) for n in self.nu]
      for l in np.arange(self.nl,step=2):
         Pv[l] = sigmoid(beta*(E[l] + self.b[l]))

      return Pv
      

   ## DBM: EO-Gradient Update
   
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
      Ph = self.__flip(d_eo)
      h0 = [np.array(np.random.rand(len(p)) < p, dtype=int) for p in Ph]
      

      ## Contrastive divergence
      h = h0
      for i in range(CDn):
         Pv = self.__flop(h)
         v  = [np.array(np.random.rand(len(p)) < p, dtype=int) for p in Pv]
         Ph = self.__flip(v)
         h  = [np.array(np.random.rand(len(p)) < p, dtype=int) for p in Ph]
         
      # Update weight gradients
      for l in range(self.nl-1):
         if l % 2 == 0:
            self.dW[l] += np.outer(v0[l], h0[l+1]) - np.outer(v[l], h[l+1])
         else:
            self.dW[l] += np.outer(h0[l], v0[l+1]) - np.outer(h[l], v[l+1])
      
      # Update bias gradients
      for l in range(self.nl):
         if l % 2 == 0:
            self.db[l] += v0[l] - v[l]
         else:
            self.db[l] += h0[l] - h[l]

   ## DBM: Mean-field variational Expectation of the Data distribution (mu)
   def __variationalExpectation(self, v, tol=1e-3, temp=1.0):
      mu = [np.zeros(n) for n in self.nu]
      mu[0] = v.copy()
      prev_mu = [x.copy() for x in mu]

      for i in xrange(10000):
         for l in xrange(1,self.nl):
            mu[l] = sigmoid(1.0/temp*(np.dot(self.W[l-1].T, mu[l-1]) +
                                      self.b[l] +
                                      (np.dot(self.W[l],mu[l+1]) if l+1 < self.nl else 0)))
            prev_mu[l] -= mu[l]
            
         if np.max([np.max(np.abs(x)) for x in prev_mu[1:]]) < tol:
            break

         prev_mu = [x.copy() for x in mu]

      return mu

   
   ## DBM: Training (EO algorithm)
   
   def __preTrainDBM(self, data, epochs, batchsize, LR=0.01, s1_alg='DAE', s1_epochs=None, s1_batchsize=None, s1_LR=None, **algopts):
      '''
      Pretrain network with Kyunghyun Cho's two-stage algorithm
      '''
      # Reset Weights and Biases
      self.__setZeroWeights()
      self.__setZeroBiases()
         
      # Arguments
      s1_epochs = epochs if s1_epochs is None else s1_epochs
      s1_batchsize = batchsize if s1_batchsize is None else s1_batchsize
      s1_LR = LR if s1_LR is None else s1_LR

      ## Stage 1.1
      sys.stdout.write('[DBM] stage 1 | alg: {} | epochs: {}\n'.format(s1_alg, s1_epochs))
      # Select layers 0,2,4...
      even_idx = np.arange(self.nl,step=2)
      even_nu = self.nu[even_idx]
      stage1net = DeepNet(even_nu)
      # Pretrain with s1_alg
      stage1net.preTrain(data, s1_epochs, s1_batchsize, LR=s1_LR, algorithm=s1_alg)

      ## Stage 1.2
      sys.stdout.write('[DBM] stage 1 | computing mean-field variational expectation of input data...\n')
      # Variational Expectation of even layers given the input data.
      # Computing on all imput samples simultaneously
      N_data = len(data)
      V_iters = 100

      mu = [np.random.random((N_data,n)) for n in even_nu]
      mu[0] = data

      for i in xrange(V_iters):
         for l in xrange(1,len(even_nu)):
            mu[l] = sigmoid(np.dot(mu[l-1], stage1net.W[l-1]) +
                            stage1net.b[l] +
                            (np.dot(mu[l+1], stage1net.W[l].T) if l+1 < len(even_nu) else 0))

      ## Stage 2
      # Train parameters with odd-even RBM (even layers are visible, odd hidden)
      data_idx = np.arange(len(data))
      d_eo = [None]*self.nl
      for s1_l, l in enumerate(even_idx):
         d_eo[l] = mu[s1_l]

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



def main():
   # Load MNIST data
   train, valid, test = mnist_loader.load_data()
   images, digits = train
   imtest, dtest = test

   # SDAE
   net_name = 'finetuned_sDAE_200_200_200_pretrain2e_finetune1e'
   layers = np.array([28*28,1000,500,250,2])
   deepnet = DeepNet(layers)
   deepnet.preTrain(images, epochs=3, batchsize=100, algorithm='DAE', LR=0.05, noise_p=0.5)
   #Dump DeepNet
   with open('pretrained_sdae_200_200_200.dump', 'w') as f:
      pickle.dump(obj=deepnet, file=f)

   sample_set = []
   for s in range(20):
      sample_set.append(deepnet.generate_DBM(images[s,:], N=40, step=1, temp=1))
   plot_digits(sample_set, fname='samples_from_digits_pretrained')

   # Loss function (PRETRAIN)
   ## Unit status
   # Compute Encoder units
   x = imtest
   z = imtest.copy()
   l = 1
   for i in np.arange(deepnet.nl-1):
      z = sigmoid(np.dot(z,deepnet.W[i]) + deepnet.b[i+1])
      
   # Compute Decoder units
   for i in np.arange(deepnet.nl-2,-1,step=-1):
      z = sigmoid(np.dot(z,deepnet.W[i].T) + deepnet.b[i])

   L_pretrain = -np.sum(x*np.log(z) + (1-x)*np.log(1-z), axis=1)
      
   deepnet.fineTune_sDAE(images, epochs=5, batchsize=100, LR=0.5, noise_p=0.5)
   # #Dump DeepNet
   with open('finetuned_sdae_200_200_200.dump', 'w') as f:
      pickle.dump(obj=deepnet, file=f)

   z = imtest.copy()
   l = 1
   for i in np.arange(deepnet.nl-1):
      z = sigmoid(np.dot(z,deepnet.W[i]) + deepnet.b[i+1])
      
   # Compute Decoder units
   for i in np.arange(deepnet.nl-2,-1,step=-1):
      z = sigmoid(np.dot(z,deepnet.W[i].T) + deepnet.b[i])

   L_finetune = -np.sum(x*np.log(z) + (1-x)*np.log(1-z), axis=1)

   print np.mean(L_pretrain)
   print np.mean(L_finetune)


   sample_set = []
   for s in range(20):
      sample_set.append(deepnet.generate_DBM(images[s,:], N=40, step=1, temp=1))
   plot_digits(sample_set, fname='samples_from_digits_finetuned')


   sys.exit(0)
   # Create DeepNet
   # layers = np.array([28*28,200,200,200])
   # #layers = np.array([28*28,100,100])
   # #layers = np.array([28*28,200])
   # deepnet = DeepNet(layers)

   # # Pre-train DeepNet as DBN
   # #deepnet.preTrain(images, epochs=10, batchsize=100, algorithm='RBM', CDn=1, LR=0.1)
   # net_name = 'dbm_200_200_200_pretrain_2step_SDAE-4e-LR05_RBM-10e-LR01-CDn20.dump'
   # deepnet.preTrain(images, epochs=4, batchsize=100, algorithm='DBM', LR=0.1, s1_epochs=2, s1_LR=0.5, CDn=20)
   # #deepnet.preTrain(images, epochs=10, batchsize=100, algorithm='DBM', LR=0.1, CDn=1)
   # # deepnet.preTrain(images, epochs=3, batchsize=100, algorithm='DAE', LR=0.05, noise_p=0.5)

   # with open('dbm_200_200_200_pretrain_2step_SDAE-4e-LR05_RBM-10e-LR01-CDn20.dump') as f:
   #    deepnet = pickle.load(f)

   # deepnet.fineTune(images)
   #   deepnet.fineTune(images)

   # with open('dbm_new_finetune_it49000.dump') as f:
   #    deepnet = pickle.load(f)         
   # Generate samples from numbers
   sample_set = []
   for s in range(20):
      sample_set.append(deepnet.generate_DBM(images[s,:], N=40, step=1, temp=1))
   plot_digits(sample_set, fname='samples_from_digits')

   # Generate samples from half numbers
   sample_set = []
   mask = np.zeros((28,28))
   mask[:,14:28] = 1
   for s in range(20):
      sample_set.append(deepnet.generate_DBM((images[s,:].reshape((28,28)) * mask).reshape(28*28), N=40, step=10))
   plot_digits(sample_set, fname='samples_from_half_digits')

   # Random data
   sample_set = []
   for s in range(20):
      sample_set.append(deepnet.generate_DBM(np.random.rand((28*28)), N=40, step=10))
   plot_digits(sample_set, fname='samples_from_random_noise')


if __name__ == '__main__':
   main()

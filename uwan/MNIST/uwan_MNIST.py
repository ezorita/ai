import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.distributions.normal import Normal

cuda_dtype  = torch.float32
show_images = False
save_images = True

class WULayer(nn.Module):
   def __init__(self, input_size, output_size, device=None, dtype=cuda_dtype):
      super(WULayer, self).__init__()
        
      # Store arguments
      self.dev = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
      self.dtype = torch.float32 if self.dev == 'cpu' else dtype

      # Initialize parameters
      self.mu = nn.Parameter(torch.zeros((output_size,input_size), dtype=self.dtype, device=self.dev))
      self.ro = nn.Parameter(torch.log(torch.exp(1.5*torch.ones((output_size, input_size), dtype=self.dtype, device=self.dev)/np.sqrt(output_size))-1))
      self.b  = nn.Parameter(torch.zeros(output_size, dtype=self.dtype, device=self.dev))
      self.W  = torch.zeros((output_size,input_size), dtype=self.dtype, device=self.dev)

   def sampleWeights(self):
      sd = torch.log(1+torch.exp(self.ro))
      W  = torch.randn(self.mu.shape, dtype=self.dtype, device=self.dev) * sd + self.mu
      return W

   def posteriorLoss(self):
      sd = torch.log(1+torch.exp(self.ro.detach()))
      return torch.mean(Normal(self.mu.detach(),sd).log_prob(self.W))

   def forward(self, x, new_weights=True):
      # Sample weight matrix
      if new_weights:
         W  = self.sampleWeights()
         self.W = W
      else:
         W = self.W

      # Compute layer output
      x = torch.mm(x, W.t()) + self.b

      return x

class RandNet(nn.Module):
   def __init__(self, layers, out_func=(lambda x: x), device=None, dtype=cuda_dtype):
      super(RandNet, self).__init__()

      # Computation device
      self.dev = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
      self.dtype = torch.float32 if self.dev == 'cpu' else dtype
      self.out_func = out_func

      # Arguments
      self.input_size  = layers[0]
      self.output_size = layers[-1]

      # Create layers
      self.layers = nn.ModuleList([WULayer(i, o).type(self.dtype) for i,o in zip(layers[:-1], layers[1:])]).to(self.dev)

   def forward(self, x, new_weights=True):
      for layer in self.layers[:-1]:
         x = torch.relu(layer(x, new_weights))

      return self.out_func(self.layers[-1](x))

   
class DetNet(nn.Module):
   def __init__(self, layers, out_func=(lambda x: x), device=None, dtype=cuda_dtype):
      super(DetNet, self).__init__()

      # Computation device
      self.dev = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
      self.dtype = torch.float32 if self.dev == 'cpu' else dtype
      self.out_func = out_func

      # Arguments
      self.input_size  = layers[0]
      self.output_size = layers[-1]

      # Create layers
      self.layers = nn.ModuleList([nn.Linear(i, o).type(self.dtype) for i,o in zip(layers[:-1], layers[1:])]).to(self.dev)

   def forward(self, x):
      for layer in self.layers[:-1]:
         x = torch.relu(layer(x))

      return self.out_func(self.layers[-1](x))

   
class UWAN(nn.Module):
   def __init__(self, encoder, decoder, z_disc, z_dist, w_dist, device=None, dtype=cuda_dtype):
      super(UWAN, self).__init__()

      # Computation device
      self.dev = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
      self.dtype = torch.float32 if self.dev == 'cpu' else dtype

      # Encoder and decoder
      self.encoder = encoder
      self.decoder = decoder
      
      # Discriminator networks
      self.z_disc = z_disc

      # Prior distributions
      self.z_dist = z_dist
      self.w_dist = w_dist

      # Data size
      self.input_dim  = encoder.input_size
      self.latent_dim = encoder.output_size
      self.output_dim = decoder.output_size
      

   def getCurrentWeights(self):
      w_vec = torch.tensor([], dtype=self.dtype, device=self.dev)
      for l in self.encoder.layers:
         w_vec = torch.cat((w_vec, l.W.reshape(-1)))
      for l in self.decoder.layers:
         w_vec = torch.cat((w_vec, l.W.reshape(-1)))
      w_vec = w_vec.reshape((-1,1))

      return w_vec

       
   def forward(self, x, new_weights=True):
      z = self.encoder(x, new_weights) if self.encoder.__class__.__name__ == 'RandNet' else self.encoder(x)
      y = self.decoder(z, new_weights) if self.decoder.__class__.__name__ == 'RandNet' else self.decoder(z)
    
      return y, z

         
   def optimize(self, train_data, test_data, epochs, batch_size, lr=1e-3, n_test_img=6, prior_std=5.0):

      # Create directories
      if not os.path.exists('{}/test_digits'.format(os.getcwd())):
         os.makedirs('{}/test_digits'.format(os.getcwd()))
      if not os.path.exists('{}/latent_space'.format(os.getcwd())):
         os.makedirs('{}/latent_space'.format(os.getcwd()))

      # Optimizer
      enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
      dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=lr)
      zds_opt = torch.optim.Adam(self.z_disc.parameters(), lr=lr)

      # Plot test input images
      test_imgs = test_data[0:n_test_img,:].type(self.dtype).view(-1,28*28)
      f, a = plt.subplots(2, n_test_img, figsize=(8, 3))
      for i in range(n_test_img):
         a[0][i].imshow(1-np.reshape(test_imgs.data.numpy()[i], (28,28)), cmap='gray')
         a[0][i].set_xticks(())
         a[0][i].set_yticks(())
      loss_text = f.text(0, 0.01, "epoch: 0, loss: 0")
      if show_images:
         plt.ion()
         plt.show()
      else:
         plt.ioff()

      # Move data to device
      train_data = train_data.type(self.dtype).to(self.dev)
      test_data = test_data.type(self.dtype).to(self.dev)
      test_imgs = test_imgs.type(self.dtype).to(self.dev)

      target_zfake = torch.zeros((2*batch_size,1), dtype=self.dtype, device=self.dev)
      target_zreal = torch.ones ((2*batch_size,1), dtype=self.dtype, device=self.dev)

      target_testfake = torch.zeros((len(test_data), 1), dtype=self.dtype, device=self.dev)
      target_testreal = torch.ones ((len(test_data), 1), dtype=self.dtype, device=self.dev)

      # Loss log
      zds_log = np.zeros(epochs)
      rec_log = np.zeros(epochs)
      pri_log = np.zeros(epochs)
      pos_log = np.zeros(epochs)
      fol_log = np.zeros(epochs)
      zds_acc_log = np.zeros(epochs)
      zds_test_log = np.zeros(epochs)
      rec_test_log = np.zeros(epochs)
      fol_test_log = np.zeros(epochs)
      zds_acc_test_log = np.zeros(epochs)

      # Epoch iteration
      epoch_idx = np.arange(train_data.shape[0])
      for e in np.arange(epochs):
         # Shuffle training data
         np.random.shuffle(epoch_idx)
         batch_idx = np.array_split(epoch_idx, train_data.shape[0]/batch_size)

         # Initialize epoch loss
         zds_blos = 0.0
         zds_bacc = 0.0
         rec_blos = 0.0
         pri_blos = 0.0
         pos_blos = 0.0
         z_blos   = 0.0

         torch.cuda.empty_cache()

         for batch_no, idx in enumerate(batch_idx):
            # Get data and targets
            data = train_data[idx,:]
            b_size = data.shape[0]
            ## DISCRIMINATOR: Latent space distribution
            # Generate current and real distribution samples
            z_real = prior_std*self.z_dist((b_size,self.latent_dim), dtype=self.dtype, device=self.dev)
            z_fake = self.encoder(data).detach()
            # Evaluate discriminator
            d = self.z_disc(torch.cat((z_fake, z_real), 0))
            dd = d.detach().to('cpu').numpy().reshape(-1)
            # Discriminator loss
            zds_loss = func.binary_cross_entropy(d, torch.cat((target_zfake[:b_size], target_zreal[:b_size]),0))
            zds_blos += float(zds_loss.detach().to('cpu').numpy())
            zds_bacc += (np.sum(dd[:b_size] < np.random.rand(b_size)) + np.sum(dd[b_size:] > np.random.rand(b_size))) / (2*b_size)
            # Compute gradients
            zds_opt.zero_grad()
            zds_loss.backward()
            # Update discriminator parameters
            zds_opt.step()

            ## AUTOENCODER
            y, z = self(data)
            dz = self.z_disc(z)
            # Reconstruction loss
            rec_loss = func.binary_cross_entropy(y, data)
            rec_blos += float(rec_loss.detach().to('cpu').numpy())
            # Latent space distribution loss
            z_loss = func.binary_cross_entropy(dz, target_zreal[:b_size])
            z_blos += float(z_loss.detach().to('cpu').numpy())
            # Weight distribution loss
            prior_loss = torch.mean(self.w_dist.logProb(self.getCurrentWeights(), dtype=self.dtype, device=self.dev))
            posterior_loss = torch.tensor([0.0], device=self.dev, dtype=self.dtype)
            for n_layers,(lenc, ldec) in enumerate(zip(self.encoder.layers, self.decoder.layers)):
               posterior_loss += .5*(lenc.posteriorLoss() + ldec.posteriorLoss())
            posterior_loss /= n_layers+1
            pri_blos += float(prior_loss.detach().to('cpu').numpy())
            pos_blos += float(posterior_loss.detach().to('cpu').numpy())
            # Total loss
            pi_i = 2**(len(batch_idx)-batch_no-1) / (2**len(batch_idx)-1)
            loss = rec_loss + pi_i*(z_loss - prior_loss + posterior_loss)
            # Compute gradients
            dec_opt.zero_grad()
            enc_opt.zero_grad()
            loss.backward()
            # Update gradients
            enc_opt.step()
            dec_opt.step()

            
         # End of epoch
         batch_no += 1
         sys.stdout.write('epoch: {}, z-Discriminator: {:.3f} ({:.3f}%), KL-weights Prior: {:.3f}, posterior: {:.3f}, Autoencoder Rec: {:.3f}, Z-fool: {:.3f}\r      \n'.format(
            e+1,
            zds_blos/batch_no,
            zds_bacc/batch_no*100,
            pri_blos/batch_no,
            pos_blos/batch_no*100,
            rec_blos/batch_no,
            z_blos/batch_no
         ))

         # Store loss record
         zds_log[e] = zds_blos/batch_no
         rec_log[e] = rec_blos/batch_no
         pri_log[e] = pri_blos/batch_no
         pos_log[e] = pos_blos/batch_no
         fol_log[e] = z_blos/batch_no
         zds_acc_log[e] = zds_bacc/batch_no

         # Test dataset
         z_real = prior_std*self.z_dist((len(test_data),self.latent_dim), dtype=self.dtype, device=self.dev)
         z_fake = self.encoder(test_data).detach()
         # Evaluate discriminator (test)
         d = self.z_disc(torch.cat((z_fake, z_real), 0)).detach()
         dd = d.to('cpu').numpy().reshape(-1)
         # Discriminator loss (test)
         zds_test_log[e] = float(func.binary_cross_entropy(d, torch.cat((target_testfake, target_testreal),0)).detach().to('cpu'))
         zds_acc_test_log[e] = (np.sum(dd[:len(test_data)] < np.random.rand(len(test_data))) + np.sum(dd[len(test_data):] > np.random.rand(len(test_data)))) / (2*len(test_data))

         # Autoencoder (test dataset)
         y, z = self(test_data)
         dz = self.z_disc(z.detach()).detach()
         # Reconstruction loss
         rec_test_log[e] = float(func.binary_cross_entropy(y.detach(), test_data).detach().to('cpu'))
         # Latent space distribution loss
         fol_test_log[e] = float(func.binary_cross_entropy(dz, target_testreal).detach().to('cpu'))

         if save_images or show_images:
            test_out, _ = self(test_imgs)
            test_out = test_out.detach().to('cpu').numpy()
            for i in range(n_test_img):
                a[1][i].imshow(1.0-np.reshape(test_out[i], (28,28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

            loss_text.set_text('epoch: {}-{}, z-Discriminator: {:.3f} ({:.3f}%), KL-weights Prior: {:.3f}, posterior: {:.3f}, Autoencoder Rec: {:.3f}, Z-fool: {:.3f}, CUDA mem: {:.2f}MB  \r'.format(
               e+1,
               batch_no+1,
               zds_blos/(batch_no+1),
               zds_bacc/(batch_no+1)*100,
               pri_blos/(batch_no+1),
               pos_blos/(batch_no+1),
               rec_blos/(batch_no+1),
               z_blos/(batch_no+1),
               torch.cuda.memory_allocated()/1e6
            ))
            f.canvas.draw()
            plt.pause(0.001)
            if save_images:
               f.savefig('test_digits/uwan_MNIST_test_digits_{}.png'.format(e+1))

            # Latent space
            z = z.detach().to('cpu').numpy()
            plt.figure()
            plt.scatter(z[:,0], z[:,1], c=test.targets, s=0.8)
            plt.savefig('latent_space/uwan_MNIST_latent_space_{}.png'.format(e+1))
            plt.close()

      return rec_log, zds_log, zds_acc_log, pri_log, pos_log, fol_log, rec_test_log, zds_test_log, zds_acc_test_log, fol_test_log


class GaussianMixture():
   def __init__(self, mu_a, sd_a, mu_b, sd_b, p_a):
      self.mu_a = mu_a
      self.mu_b = mu_b
      self.sd_a = sd_a
      self.sd_b = sd_b
      self.p_a  = p_a

   def __call__(self, sample_size, device='cpu', dtype=cuda_dtype):
      a_pos = (torch.rand(sample_size, device=device, dtype=dtype) < self.p_a).type(dtype)
      a_mix = self.mu_a + self.sd_a*torch.randn(sample_size, device=device).type(dtype)
      b_mix = self.mu_b + self.sd_b*torch.randn(sample_size, device=device).type(dtype)
      return a_pos * a_mix + (1-a_pos) * b_mix

   def logProb(self, samples, device='cpu', dtype=cuda_dtype):
      mix_a = Normal(self.mu_a, self.sd_a).log_prob(samples) + torch.log(torch.tensor([self.p_a], dtype=dtype).to(device))
      mix_b = Normal(self.mu_b, self.sd_b).log_prob(samples) + torch.log(torch.tensor([1-self.p_a], dtype=dtype).to(device))
      return torch.logsumexp(torch.cat([mix_a.view(-1,1), mix_b.view(-1,1)], dim=1),dim=1)
  
  

if __name__ == "__main__":
   # Load dataset
   import torchvision
   train = torchvision.datasets.MNIST('./', train=True, download=True, transform=torchvision.transforms.ToTensor())
   test = torchvision.datasets.MNIST('./', train=False, download=True, transform=torchvision.transforms.ToTensor())

   train_data = train.data.view(-1,28*28).type(torch.float32)/255.0
   test_data  = test.data.view(-1,28*28).type(torch.float32)/255.0

   # Instantiate UWAN
   input_dim  = 28*28
   latent_dim = 10
   
   layers  = [input_dim, 1000, 1000, latent_dim]
   encoder = RandNet(layers)
   decoder = RandNet(np.flip(layers), out_func=torch.sigmoid)
   z_discr = DetNet([latent_dim, 1000, 1000, 1], out_func=torch.sigmoid)
   uwan = UWAN(encoder, decoder, z_discr, GaussianMixture(0, 1, 0, 1, 1.0), GaussianMixture(0, .01, 0, 2, 0.7))

   logs = uwan.optimize(train_data, test_data, 110, 128)
   rec_log, zds_log, zds_acc_log, pri_log, pos_log, fol_log, rec_test_log, zds_test_log, zds_acc_test_log, fol_test_log = logs

   torch.save(uwan.state_dict(), 'uwan_MNIST_parameters.torch')
   np.save('uwan_MNIST_loss_log.npy', np.array(logs))

   uwan.eval()

   # Train/Test loss
   plt.figure()
   plt.plot(rec_log)
   plt.plot(rec_test_log)
   plt.legend(['Train', 'Test'])
   plt.savefig('uwan_MNIST_train_loss.png')

   # Number reconstruction
   samples = 30
   iters   = 30
   step    = 1

   # Force 0 on half of training samples
   test_samples = test_data[0:samples, :].clone().to(uwan.dev)
   test_samples[:, 392:784] = torch.zeros(samples,392).to(uwan.dev)
   iter_samples = test_samples.clone()

   image = np.zeros((28*samples, 28*(iters+2)))
   image[:, 0:28] = test_data[0:samples,:].view(-1,28).detach().numpy()
   image[:, 28:56] = test_samples.to('cpu').view(-1,28).detach().numpy()

   # Iterate
   for it in np.arange(iters)+2:
      # Process image
      for j in np.arange(step):
         iter_samples[:, 0:392] = test_samples[:, 0:392]
         iter_samples, _ = uwan(iter_samples)
         iter_samples = iter_samples.detach()
      # Save image
      image[:, (it*28):((it+1)*28)] = iter_samples.to('cpu').view(-1,28).numpy()

   plt.figure(figsize=(10,10))
   plt.imshow(1-image, cmap='gray')
   plt.savefig('uwan_MNIST_reconstruct.png')
   




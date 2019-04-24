import sys, os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import pdb

# Add parent folder to PATH
sys.path.insert(0,'{}/..'.format(os.getcwd()))
from uwan import UWAN, RandNet, DetNet, GaussianMixture

if __name__ == "__main__":
   # Load dataset
   d = np.loadtxt('sc_mouse_TF.txt.gz', delimiter='\t')
   # Take 10% as test samples
   np.random.seed(10)
   test_idx = np.random.choice(np.arange(d.shape[0]), int(d.shape[0]*.1), replace=False)
   # Select test/train datasets
   train_data = torch.tensor(np.delete(d, test_idx, axis=0), dtype=torch.float32)
   test_data  = torch.tensor(d[test_idx, :], dtype=torch.float32)

   # Normalize data
   train_data = train_data/torch.sum(train_data, 1).reshape(-1,1)*train_data.shape[1]
   test_data  = test_data/torch.sum(test_data, 1).reshape(-1,1)*train_data.shape[1]

   # Instantiate UWAN
   input_dim  = train_data.shape[1]
   latent_dim = 4
   
   layers  = [input_dim, 200, 200, latent_dim]
   encoder = RandNet(layers)
   decoder = RandNet(np.flip(layers), out_func=nn.functional.softmax)
   z_discr = DetNet([latent_dim, 200, 200, 1], out_func=torch.sigmoid)
   uwan = UWAN(encoder, decoder, z_discr, GaussianMixture(0, 1, 0, 1, 1.0), GaussianMixture(0, .01, 0, 2, 0.7))

   logs = uwan.optimize(train_data, test_data, 100, 128, lr=1e-4, rec_loss_func=nn.functional.mse_loss)
   rec_log, zds_log, zds_acc_log, pri_log, pos_log, fol_log, rec_test_log, zds_test_log, zds_acc_test_log, fol_test_log = logs

   torch.save(uwan.state_dict(), 'uwan_parameters.torch')
   np.save('uwan_loss_log.npy', np.array(logs))

   uwan.eval()

   # Train/Test loss
   plt.figure()
   plt.plot(rec_log)
   plt.plot(rec_test_log)
   plt.legend(['Train', 'Test'])
   plt.savefig('uwan_train_loss.png')

   # Latent space
   y, z = uwan(test_data.to(uwan.dev))
   z = z.detach().to('cpu').numpy()
   plt.figure()
   plt.scatter(z[:,0], z[:,1], s=0.8)
   plt.savefig('uwan_latent_space.png')

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:04:17 2021

@author: rodol
"""

import torch
import torch.utils.data
import torch.nn.functional as F
import timeit
import numpy as np

'FIX THE SEED -Controlling sources of randomness (REPRODUCIBILITY)' 
def set_seed():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)

      

class Encoder(torch.nn.Module): # M x 2 x 256 (Y, X)
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = torch.nn.Sequential(
              torch.nn.Conv1d(2, 32, kernel_size=5, stride=2, padding=2), # M x 32 x 128
              torch.nn.Tanh())
        
        # Dense block 1
        self.dense11 = torch.nn.Sequential(torch.nn.BatchNorm1d(32),
              torch.nn.Conv1d(32, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense12 = torch.nn.Sequential(torch.nn.BatchNorm1d(48),
              torch.nn.Conv1d(48, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense13 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense14 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        
        self.layer2 = torch.nn.Sequential(
              torch.nn.Conv1d(96, 48, kernel_size=5, stride=2, padding=2), # M x 48 x 64
              torch.nn.Tanh())
        
        # Dense block 2
        self.dense21 = torch.nn.Sequential(torch.nn.BatchNorm1d(48),
              torch.nn.Conv1d(48, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense22 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense23 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense24 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        self.layer3 = torch.nn.Sequential(
              torch.nn.Conv1d(112, 56, kernel_size=5, stride=2, padding=2), # M x 56 x 32
              torch.nn.Tanh())
        
        # Dense block 3
        self.dense31 = torch.nn.Sequential(torch.nn.BatchNorm1d(56),
              torch.nn.Conv1d(56, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense32 = torch.nn.Sequential(torch.nn.BatchNorm1d(72),
              torch.nn.Conv1d(72, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense33 = torch.nn.Sequential(torch.nn.BatchNorm1d(88),
              torch.nn.Conv1d(88, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense34 = torch.nn.Sequential(torch.nn.BatchNorm1d(104),
              torch.nn.Conv1d(104, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
                
        self.layer4 = torch.nn.Sequential(
              torch.nn.Conv1d(120, 60, kernel_size=5, stride=2, padding=2), # M x 60 x 16
              torch.nn.Tanh())
        
        # Dense block 4
        self.dense41 = torch.nn.Sequential(torch.nn.BatchNorm1d(60),
              torch.nn.Conv1d(60, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense42 = torch.nn.Sequential(torch.nn.BatchNorm1d(76),
              torch.nn.Conv1d(76, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense43 = torch.nn.Sequential(torch.nn.BatchNorm1d(92),
              torch.nn.Conv1d(92, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense44 = torch.nn.Sequential(torch.nn.BatchNorm1d(108),
              torch.nn.Conv1d(108, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        
        self.layer5 = torch.nn.Sequential(
              torch.nn.Conv1d(124, 256, kernel_size=5, stride=2, padding=2), # M x 256 x 8
              torch.nn.Tanh())        
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(256*8, 32))
        
       
    
    # def dense_ed(self,x):
    #     for idx in range(4):
    #         out = self.bn(x.shape[1])(x)
    #         out = self.tan(out)
    #         out = self.conv(out.shape[1],self.channel_rate, kernel_size=3, stride=1, padding=1)(out)
    #         x   = self.conc((out,x),dim=1)
    #     return x           
                    
    def forward(self, x):
        x = self.layer1(x)
        out = self.dense11(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense12(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense13(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense14(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer2(x)
        out = self.dense21(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense22(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense23(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense24(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer3(x)
        out = self.dense31(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense32(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense33(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense34(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer4(x)
        out = self.dense41(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense42(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense43(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense44(x)
        x  = torch.cat((out,x),dim=1)            
        
        out = self.layer5(x)
        out = out.view(-1, 256*8) # M x 256*8
        out = self.fc(out)
        out = out.view(-1, 32, 1) # M x 64 x 1
        return out  # M x Z_dim x 1


class Discriminator(torch.nn.Module): # M x 2 x 256 (Y, X)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = torch.nn.Sequential(
              torch.nn.Conv1d(2, 32, kernel_size=5, stride=2, padding=2), # M x 32 x 128
              torch.nn.BatchNorm1d(32),
              torch.nn.Tanh())
        self.layer2 = torch.nn.Sequential(
              torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # M x 64 x 64
              torch.nn.BatchNorm1d(64),
              torch.nn.Tanh())
        self.layer3 = torch.nn.Sequential(
              torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # M x 128 x 32
              torch.nn.BatchNorm1d(128),
              torch.nn.Tanh())
        self.layer4 = torch.nn.Sequential(
              torch.nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # M x 256 x 16
              torch.nn.BatchNorm1d(256),
              torch.nn.Tanh())
        self.fc = torch.nn.Sequential(
               torch.nn.Linear(256*16, 1)) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 256*16) # M x 256*16
        out = self.fc(out)
        return out # M x 1

class Decoder(torch.nn.Module): # M x (32) x 1 (Z)
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(288, 64, kernel_size = 4, stride = 1, padding = 0), # M x 64 x 4
              torch.nn.Tanh())
        
        # Dense block 1
        self.dense11 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense12 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense13 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense14 = torch.nn.Sequential(torch.nn.BatchNorm1d(112),
              torch.nn.Conv1d(112, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        
        
        self.layer2 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(128, 64, kernel_size = 4, stride = 2, padding = 1), # M x 64 x 8
              torch.nn.Tanh())
        
        # Dense block 2
        self.dense21 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense22 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense23 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense24 = torch.nn.Sequential(torch.nn.BatchNorm1d(112),
              torch.nn.Conv1d(112, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        self.layer3 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(128, 64, kernel_size = 4, stride = 2, padding = 1), # M x 64 x 16
              torch.nn.Tanh())
        
        # Dense block 3
        self.dense31 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense32 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense33 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense34 = torch.nn.Sequential(torch.nn.BatchNorm1d(112),
              torch.nn.Conv1d(112, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        self.layer4 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(128, 64, kernel_size = 4, stride = 2, padding = 1), # M x 64 x 32
              torch.nn.Tanh())
        
        # Dense block 4
        self.dense41 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense42 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense43 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense44 = torch.nn.Sequential(torch.nn.BatchNorm1d(112),
              torch.nn.Conv1d(112, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        self.layer5 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(128, 64, kernel_size = 4, stride = 2, padding = 1), # M x 64 x 64
              torch.nn.Tanh())
        
        # Dense block 5
        self.dense51 = torch.nn.Sequential(torch.nn.BatchNorm1d(64),
              torch.nn.Conv1d(64, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense52 = torch.nn.Sequential(torch.nn.BatchNorm1d(80),
              torch.nn.Conv1d(80, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense53 = torch.nn.Sequential(torch.nn.BatchNorm1d(96),
              torch.nn.Conv1d(96, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())  
        self.dense54 = torch.nn.Sequential(torch.nn.BatchNorm1d(112),
              torch.nn.Conv1d(112, 16, kernel_size=3, stride=1,  padding=1), 
              torch.nn.ReLU())
        
        self.layer6 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(128, 64, kernel_size = 4, stride = 2, padding = 1), # M x 64 x 128
              torch.nn.Tanh())
        self.layer7 = torch.nn.Sequential(
              torch.nn.ConvTranspose1d(64, 1, kernel_size = 4, stride = 2, padding = 1)) # M x 1 x 256

    def forward(self, z, x):
        x = torch.reshape(x,(x.shape[0],256,1))
        z = torch.cat((z,x),dim=1)
        
        x = self.layer1(z)
        out = self.dense11(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense12(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense13(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense14(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer2(x)
        out = self.dense21(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense22(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense23(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense24(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer3(x)
        out = self.dense31(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense32(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense33(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense34(x)
        x  = torch.cat((out,x),dim=1)            
        
        x = self.layer4(x)
        out = self.dense41(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense42(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense43(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense44(x)
        x  = torch.cat((out,x),dim=1)
        
        x = self.layer5(x)
        out = self.dense51(x)
        x   = torch.cat((out,x),dim=1)
        out = self.dense52(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense53(x)
        x  = torch.cat((out,x),dim=1)
        out = self.dense54(x)
        x  = torch.cat((out,x),dim=1)
        
        out = self.layer6(x)
        out = self.layer7(out)
        
        return out # M x 1 x 256


class CADGM_HD:
    # Initialize the class
    def __init__(self, Y, X, lam = 1.5, beta = 0.):  
      
        #Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.encoder = Encoder().cuda()
            self.decoder = Decoder().cuda()
            self.discriminator = Discriminator().cuda()
        else:
            self.dtype_double = torch.FloatTensor
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.discriminator = Discriminator()

        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        Y = (Y - self.Ymean)/(self.Ystd + 1e-8)
        X = (X - self.Xmean)/self.Xstd
        self.Z_dim = 32
        self.lam = lam 
        self.beta = beta        
        
        # Define PyTorch dataset
        self.Y = torch.from_numpy(Y).type(self.dtype_double) # num_images x num_channels x num_pixels_x x num_pixels_y
        self.X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_channels x num_pixels_x x num_pixels_y
        
        # Define the optimizer
        self.optimizer_G = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4, betas=(0.9, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.discriminator.parameters()), lr=1e-4, betas=(0.9, 0.999))
     
        
    def net_encoder(self, Y, X):  # M x 1 x 256, M x 1 x 256
        T = torch.cat((Y, X), dim = 1)  # M x 2 x 256
        Z =  self.encoder(T)
        return Z  # M x 32 x 1
    

    def net_decoder(self, Z, X):  # M x 32 x 1, M x 1 x 256
        #T = torch.cat((Z, X), dim = 1)  # M x 64 x 1
        Y =  self.decoder(Z,X)  
        return Y  # M x 1 x 256

    def net_discriminator(self, Y, X):  # M x 1 x 256, M x 1 x 256
        T = torch.cat((Y, X), dim = 1)  # M x 2 x 256
        Y =  self.discriminator(T)  
        return Y  # M x 1

    def compute_G_loss(self, Y, X, Z): # M x 1 x 256, M x 1 x 256, M x 32 x 1
        # Broadcast the label x as a vector having same size with latent variable
        # Decoder: p(y|x,z)
        Y_pred = self.net_decoder(Z, X)
        # Encoder: q(z|x,y)
        Z_pred = self.net_encoder(Y_pred, X)
        # Discriminator loss
        T_pred = self.net_discriminator(Y_pred, X)
        # Compute the KL-divergence
        KL = torch.mean(T_pred)
        # Entropic regularizaiton
        log_q = - torch.mean((Z - Z_pred)**2)
        # Reconstruction loss
        log_p = torch.mean((Y - Y_pred)**2)
        # Generator loss
        loss = KL + (1 - self.lam)*log_q + self.beta * log_p
        return loss, KL, (1 - self.lam)*log_q, log_p

    def compute_D_loss(self, Y, X, Z):
        # Decoder: p(y|x,z)
        Y_pred = self.net_decoder(Z, X)
        #xxx = X.repeat(1, 1, 256)
        # Discriminator loss
        T_real = torch.sigmoid(self.net_discriminator(Y, X))
        T_fake = torch.sigmoid(self.net_discriminator(Y_pred, X))

        loss = - torch.mean(torch.log(1.0 - T_real + 1e-8) + torch.log(T_fake + 1e-8))
        
        return loss, T_real[0,0], T_fake[0,0]
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self, Y, X, N_batch):
        N = Y.shape[0]
        idx = torch.randperm(N)[0: N_batch]
        Y_batch = Y[idx,:,:]
        X_batch = X[idx,:,:]        
        return Y_batch, X_batch
    
    
    # Trains the model by minimizing the loss
    def train(self, nIter = 10000, batch_size = 64):
        
        set_seed()
        
        save_loss_D = []
        save_loss_G = []
        start_time = timeit.default_timer()
        for it in range(nIter):
            
            # Fetch mini-batch
            Y_batch, X_batch = self.fetch_minibatch(self.Y, self.X, batch_size)
            Z_batch = torch.randn(batch_size, self.Z_dim, 1).type(self.dtype_double)
            
            # Reset gradients for next step
            for k1 in range(1):
              self.optimizer_D.zero_grad()
              # Discriminator loss
              loss_D, real, fake = self.compute_D_loss(Y_batch, X_batch, Z_batch)
              loss_D.backward()
              # Train op for discriminator
              self.optimizer_D.step()

            for k2 in range(2):
              self.optimizer_G.zero_grad()
              # Generator loss
              loss_G, KL, log_q, log_p = self.compute_G_loss(Y_batch, X_batch, Z_batch)
              loss_G.backward()
              # Train op for generator
              self.optimizer_G.step() 
            
            save_loss_D.append(loss_D.cpu().data.numpy())
            save_loss_G.append(loss_G.cpu().data.numpy())
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, KL: %.3e, Z_recon: %.3e, U_recon: %.3e, T_loss: %.3e, Real: %.3e, Fake: %.3e, Time: %.2f' % 
                      (it, 
                       KL.cpu().data.numpy(), 
                       log_q.cpu().data.numpy(),
                       log_p.cpu().data.numpy(),
                       loss_D.cpu().data.numpy(),
                       real.cpu().data.numpy(),
                       fake.cpu().data.numpy(),
                       elapsed))
                start_time = timeit.default_timer()
        
        return save_loss_D, save_loss_G
  
    
   # Evaluates predictions at test points    
    def predict(self, X_star):
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
        else:
            self.dtype_double = torch.FloatTensor

        X_star = (X_star - self.Xmean)/self.Xstd 
        X_star = torch.from_numpy(X_star).type(self.dtype_double)
        Z = torch.randn(X_star.shape[0], self.Z_dim, 1).type(self.dtype_double)
        #xx = X_star.repeat(1, self.Z_dim, 1)
        Y_star = self.net_decoder(Z, X_star)
        Y_star = Y_star.cpu().data.numpy()
        # De-normalize the data
        Y_star = Y_star * (self.Ystd + 1e-8) + self.Ymean
        return Y_star

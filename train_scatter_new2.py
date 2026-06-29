#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:17:19 2026

@author: kishan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_scatter2 import *
from utilities3 import count_params, LpLoss, UnitGaussianNormalizer


import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--in_channels', type=int, default=4, help='')
parser.add_argument('--out_channels', type=int, default=2, help='')
parser.add_argument('--modes', type=int, default=64, help='')
parser.add_argument('--num_layers', type=int, default=6, help='')
parser.add_argument('--width', type=int, default=256, help='')
parser.add_argument('--batch', type=int, default=16, help='')
parser.add_argument('--lr', type=float, default=0.00001, help='')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--step_size', type=int, default=100, help='')
parser.add_argument('--gamma', type=float, default=0.5, help='')
parser.add_argument('--L',  metavar='N',  type=int,  nargs='*',  default=[1,2,3,4,5], help='')
parser.add_argument('--M',  metavar='N',  type=int,  nargs='*',  default=[-10,-20,-30,-40,-50], help='') #[-10,-20,-30,-40,-50]
parser.add_argument('--device', type=str, default='cuda:3', help='')
args  = parser.parse_args()


data =[]
#with open('input.npy', 'wb') as f:
#    data = np.load(f)

data = np.load('data/compressed_dataset.npz')

#print(data['data'].shape)

data = data['data']

batch_size = args.batch
learning_rate = args.lr
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma
# Using more modes or width doesn't help.
modes1 = args.modes  # At most nt
width = args.width  # Max = 190 in NVIDIA GeForce RTX 2080 Ti, otherwise OOM
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device(args.device)

ntrain = 1000
#ntest = 1000


l = args.L #[2,4] # [1,2,3,5,7,9,12,14,16]
m = args.M # [3,5] #[2,5,8,10,13,15,17,19]

x_train = data[:,0:3,:]  # sselect x,y, q(x,y)
u_train = data[:,4,:] # real value of output
#u_train = data[:,:,5]

print(x_train.shape)
print(u_train.shape)

# Get x,y coordinates 
X = x_train[0]

x = X[:,0]     
y = X[:,1] 


x_train = np.transpose(x_train, axes=(0, 2, 1))
#u_train = np.transpose(u_train, axes=(0, 2, 1))
#np.moveaxis(x_train, 0, -1)
#np.moveaxis(u_train, 0, -1)

#print(x_train.shape)
#print(u_train.shape)


# Computing Laplace
X = x_train[0]

x = X[:,0]     
y = X[:,1] 

A,Ainv = generate_eigen_basis(x,y,2*np.max(x),l,2*np.max(y),m)

#print(np.isclose(np.linalg.det(A), 0.0))

A = torch.from_numpy(A.astype(np.float32)).to(device)
Ainv =torch.from_numpy(Ainv.astype(np.float32)).to(device)

x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
u_train = torch.from_numpy(u_train.astype(np.float32)).to(device)
x_shape = x_train.shape


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)


w1,m1 = np.shape(A)

#m1 =3
model = Scatter_Net2( 3,1,3, m1, w1, A, Ainv, A, Ainv).to(device)

sx1 = np.shape(x)

# Remove weight_decay doesn't help.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#start_time = default_timer()
myloss = LpLoss(size_average=False)
#y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    #t1 = default_timer()
    #train_l2 = 0
    train_mse = 0
    i = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) # x.cuda(), y.cuda()
        
        #print(i)
        i = i +1
        optimizer.zero_grad()
        out = model(x)
        out = out.squeeze()
        out_shape = out.shape
        #print(out_shape)
        #out = out.reshape(batch_size, x_shape[1])
        #out = y_normalizer.decode(out)
        #y = y_normalizer.decode(y)
        
        mse = F.mse_loss(out.view(out_shape[0], -1), y.view(out_shape[0], -1), reduction='mean')
        mse.backward()
        
        print("Training Error: " + str(mse.cpu().item()))
        
        #loss = myloss(out.view(out_shape[0],-1), y.view(out_shape[0],-1))
        # loss.backward()

        optimizer.step()
        train_mse += mse.item()
        #train_l2 += loss.item()

    #scheduler.step()

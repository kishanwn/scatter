#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:58:00 2026

@author: kishan
"""

import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F



## ReQU activation
def requ(x):
    return x**2

def laplace_single_phi(x,L,m):
    return np.sqrt(2/L)*np.sin(m*np.pi*x/L)

def laplace_single_eigen(L,m):
    return (m*np.pi/L)**2

def laplace_2D_phi(x,y,L1,m1,L2,m2):
    return (2.0/np.sqrt(L1*L2))*np.sin(m1*np.pi*x/L1)*np.sin(m2*np.pi*y/L2)

def laplace_2D_eigen(L1,m1,L2,m2):
    return (m1*np.pi/L1)**2 + (m2*np.pi/L2)**2


def generate_eigen_basis(x,y,L1,l, L2, m):
    sx = np.shape(x)
    sl = np.shape(l)
    sm = np.shape(m)
    #print(np.shape(x))
    #print(np.shape(l))
    A = np.zeros((sx[0], sl[0]*sm[0]))
    #print(m[-1])
    #print(l[-1])
    for i in range(sx[0]):
        for j in range(sl[0]):
            for k in range(sm[0]):
                A[i,j*sm[0] + k] = laplace_2D_phi(x[i], y[i], L1, l[j], L2, m[k])         
    Apinv = np.linalg.pinv(A)
    return A, Apinv
    

   

def ReQU(x):
    return torch.max(x,0)**2
        
class Conv_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes1  , LAP_MATRIX, LAP_INVERSE):
        
        super(Conv_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes1 = LAP_MATRIX.shape[1]
        self.LAP_MATRIX = LAP_MATRIX
        self.LAP_INVERSE = LAP_INVERSE

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
        
        #print(self.weights1)             
        ################################################################
        # Encode
        ################################################################
        #print(x.size())
        x =  x.permute(0, 2, 1)
        #print(x.size())
        #print(self.LAP_INVERSE.size())
        x = self.LAP_INVERSE @ x  
        x = x.permute(0, 2, 1)
        #print(x.size())
        #print(self.weights1.size())
        ################################################################
        # Approximator
        ################################################################
        #x = torch.einsum("bix,iox->box", x[:,:,:self.modes1], self.weights1)
        x = torch.einsum("bix,iox->box", x[:,:], self.weights1)
        ################################################################
        # Decode
        ################################################################
        #x =  x @ self.LAP_MATRIX[:,:self.modes1].T
        x =  x @ self.LAP_MATRIX.T
        
        return x
    
        
class Scatter_Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers,  modes, width, LAP_MATRIX, LAP_INVERSE, LAP_MATRIX_LAST, LAP_INVERSE_LAST):
        super(Scatter_Net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.padding = 2 # pad the domain if input is non-periodic
        
        self.LAP_MATRIX = LAP_MATRIX
        self.LAP_INVERSE = LAP_INVERSE
        self.LAP_MATRIX_LAST = LAP_MATRIX_LAST
        self.LAP_INVERSE_LAST = LAP_INVERSE_LAST

        self.fc0 = nn.Linear(input_dim, self.width) 

        self.layers_conv_block = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv_block.append(Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE ))
        
        self.layers_conv_block.append(Conv_block(self.width, output_dim, self.modes1, self.LAP_MATRIX_LAST, self.LAP_INVERSE_LAST ))
        
        self.layers_conv1d = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv1d.append(nn.Conv1d(self.width, self.width, 1))
        

        
        '''
        self.conv0 = Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE )
        self.conv1 = Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE )
        self.conv2 = Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE )
        self.conv3 = Conv_block(self.width, 1, self.modes1, self.LAP_MATRIX_LAST, self.LAP_INVERSE_LAST )
        '''
        
        '''
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        '''
        #self.w3 = nn.Conv1d(self.width, self.width, 1)

        #self.fc1 = nn.Linear(self.width, 128)
        #self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        y = x
        x = self.fc0(y)
        x = x.permute(0, 2, 1)
        
        for i in range(self.num_layers-1):
            x1 = self.layers_conv_block[i](x)
            x2 = self.layers_conv1d[i](x)
            x = x1 + x2
            x = F.gelu(x)
        
        x = self.layers_conv_block[self.num_layers-1](x)
        x = x.permute(0, 2, 1)
        
        return x


class Scatter_Net2(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers,  modes, width, LAP_MATRIX, LAP_INVERSE, LAP_MATRIX_LAST, LAP_INVERSE_LAST):
        super(Scatter_Net2, self).__init__()

        self.modes1 = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.padding = 2 # pad the domain if input is non-periodic
        
        self.LAP_MATRIX = LAP_MATRIX
        self.LAP_INVERSE = LAP_INVERSE
        self.LAP_MATRIX_LAST = LAP_MATRIX_LAST
        self.LAP_INVERSE_LAST = LAP_INVERSE_LAST

        self.fc0 = nn.Linear(input_dim, self.width) 

        self.layers_conv_block = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv_block.append(Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE ))
        
        self.layers_conv_block.append(Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX_LAST, self.LAP_INVERSE_LAST ))
        
        self.layers_conv1d = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv1d.append(nn.Conv1d(self.width, self.width, 1))
        


        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, output_dim)

    def forward(self, x):
        
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        y = x
        x = self.fc0(y)
        x = x.permute(0, 2, 1)
        
        for i in range(self.num_layers-1):
            x1 = self.layers_conv_block[i](x)
            x2 = self.layers_conv1d[i](x)
            x = x1 + x2
            x = F.gelu(x)
        
        x = self.layers_conv_block[self.num_layers-1](x)
        x = x.permute(0, 2, 1)
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class Scatter_Net3(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers,  modes, width, LAP_MATRIX, LAP_INVERSE, LAP_MATRIX_LAST, LAP_INVERSE_LAST):
        super(Scatter_Net3, self).__init__()

        self.modes1 = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.padding = 2 # pad the domain if input is non-periodic
        
        self.LAP_MATRIX = LAP_MATRIX
        self.LAP_INVERSE = LAP_INVERSE
        self.LAP_MATRIX_LAST = LAP_MATRIX_LAST
        self.LAP_INVERSE_LAST = LAP_INVERSE_LAST

        self.fc0 = nn.Linear(input_dim, self.width) 

        self.layers_conv_block = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv_block.append(Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX, self.LAP_INVERSE ))
            
        self.batch_norm = nn.ModuleList()
        for i in range(num_layers-1):
            self.batch_norm.append(nn.BatchNorm1d(self.width))
        
        self.layers_conv_block.append(Conv_block(self.width, self.width, self.modes1, self.LAP_MATRIX_LAST, self.LAP_INVERSE_LAST ))
        
        self.layers_conv1d = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers_conv1d.append(nn.Conv1d(self.width, self.width, 1))
        


        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, output_dim)

    def forward(self, x):
        
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        y = x
        x = self.fc0(y)
        x = x.permute(0, 2, 1)
        
        for i in range(self.num_layers-1):
            x1 = self.layers_conv_block[i](x)
            x2 = self.layers_conv1d[i](x)
            x = x1 + x2
            x = F.gelu(x)
            x = self.batch_norm[i](x)
        
        x = self.layers_conv_block[self.num_layers-1](x)
        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

    
if __name__ == "__main__":
    x = np.linspace(0.0, 1.0, num=5)
    y = np.linspace(0.0, 1.0, num=5)
    l = [1,2,3]
    m = [2,5,8]
    
    
    
    A,Ainv = generate_eigen_basis(x,y,l,m)
    print(A)
    Al,Ainvl = generate_eigen_basis(x,y,[l[0]],[m[0]])
    print(A)
    print(Ainv)
    print(Al)
    print(Ainvl)
    
    w1,m1 = np.shape(A)
    print(w1)
    print(m1)
    model = Scatter_Net( m1, w1, A, Ainv, A, Ainv)
    
    batch = 100
    sx1 = np.shape(x)
    x = torch.rand(batch, sx1[0] ,4)
    u = torch.rand(batch, sx1[0] ,4)
    print(np.shape(x))
    y = model(x,u)
    print(y.size())
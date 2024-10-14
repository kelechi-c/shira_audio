'''
minimal implementation of CLAP(Contrastive Language-Audio Pretraining) from the paper.
THe archtecture is behind the working principles of this library
'''

from torch import nn

class ClapModel(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
    
class LinearProjection(nn.Module):
    def __init__(self, input_dim, out_dim):
        self.linear_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim*2, out_dim)
        )
        
    def forward(self, x):
        proj_output = self.linear_projection(x)
        
        return proj_output
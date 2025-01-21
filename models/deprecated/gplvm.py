import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.contrib.gp as gp
import pyro.ops.stats as stats
from pyro.infer.util import torch_backward, torch_item
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# class GPLVM(PyroModule):
#     def __init__(self, X, y, kernel, num_latents, Xu, noise=torch.tensor(0.01), device=torch.device('cpu')):
#         super().__init__()
#         self.device = device
#         self.num_obs = y.size(1)
#         self.num_latents = num_latents
#         self.num_inducing = Xu.shape[0]
#         self.kernel = kernel
#         self.noise = noise
        
#         self.X = PyroParam("X", torch.zeros(self.num_obs, self.num_latents, device=self.device))
#         #self.X = PyroParam("X", torch.tensor(X,device=self.device))
        



def test():
    URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    df = pd.read_csv(URL, index_col=0)
    print("Data shape: {}\n{}\n".format(df.shape, "-" * 21))
    print("Data labels: {}\n{}\n".format(df.index.unique().tolist(), "-" * 86))
    print("Show a small subset of the data:")
    print(df.head())
    
    y = torch.tensor(df.values, dtype=torch.get_default_dtype()).t()
    #print(y.shape)
    #print(y.size(1))
    
    nLatent = 2
    kernel = gp.kernels.RBF(input_dim=nLatent, lengthscale=torch.ones(nLatent))
    
    # Setup mean of the prior over X
    X_prior_mean = torch.zeros(y.size(1), nLatent)
    X = torch.nn.Parameter(X_prior_mean.clone())
    Xu = stats.resample(X_prior_mean,32)
    gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01), jitter=1e-5)
    gplvm.X = PyroSample(dist.Normal(X_prior_mean, 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)
    
    optimizer = torch.optim.Adam(gplvm.parameters(),lr=0.05)
    loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss
    
    num_steps = 3000
    losses = []
    
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(gplvm.model, gplvm.guide)
        torch_backward(loss)
        return loss
    
    for i in range(num_steps):
        loss = optimizer.step(closure)
        if i % 10 == 0:
            writer.add_scalar("Loss/train", loss, i)
            writer.flush()
        if i % 100 == 0:
            print(f"Iteration {i}, ELBO loss: {loss}")
            losses.append(loss)
    
    gplvm.mode = "guide"
    X = gplvm.X
    
    X = gplvm.X_loc.detach().numpy()

    fig = plt.figure()
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("Latent Dim 1", fontsize=14)
    plt.ylabel("Latent Dim 2", fontsize=14)
    plt.title("GPLVM on Single-Cell qPCR data", fontsize=16)
    
    writer.add_figure("Latent Spaces",fig)
    

    
    return
    
    
    
    
if __name__ == '__main__':
    test()
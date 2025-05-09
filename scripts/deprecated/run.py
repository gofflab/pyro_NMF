#!/usr/bin/env -S python3 -u
#%%
import torch
import pyro
import pyro.optim
from models.model import ProbNMFModel
from models.guide import CoGAPSGuide
from models.utils import generate_test_data, generate_structured_test_data
import pyro.poutine as poutine
from datetime import datetime
from pyro.infer.autoguide import AutoNormal

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Import tensorboard & setup writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Define the dimensions of the data and the latent space
num_genes = 500
num_samples = 100
num_patterns = 20

# Set device
# if torch.backends.cuda.is_available():
#     device=torch.device('cuda')
    
if torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device=torch.device('cpu')

#device=torch.device('cpu')

# Generate synthetic test data
D, A_true, P_true = generate_structured_test_data(num_genes, num_samples, num_patterns)

# Move data to device
D = D.to(device)

#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 500

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.05})

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
#model = CoGAPSModel(D, num_patterns, device=device)
model = ProbNMFModel(D, num_patterns, device=device)

#%%
# Inspect model
pyro.render_model(model, model_args=(D,), 
                render_params=True,
                render_distributions=True,
                #render_deterministic=True,
                filename="model.pdf")

#%% Logging model
#model.eval()
#writer.add_graph(model,D)
#model.train()

#%% Instantiate the guide
# guide = CoGAPSGuide(D, num_patterns, device=device)
guide = AutoNormal(model)

# The code snippet `# #%% Define the inference algorithm
# # svi = pyro.infer.SVI(model=model,
# #                     guide=guide,
# #                     optim=optimizer,
# #                     loss=loss_fn)` is defining the Stochastic Variational Inference (SVI)
# algorithm in Pyro.
#%% Define the inference algorithm
svi = pyro.infer.SVI(model=model,
                    guide=guide,
                    optim=optimizer,
                    loss=loss_fn)

# #%% Trace
# #trace = poutine.trace(model(D)).get_trace()
# #trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
# #print(trace.format_shapes())

#%% Run inference
for step in range(num_steps):
    loss = svi.step(D)
    if step % 10 == 0:
        writer.add_scalar("Loss/train", loss, step)
        writer.flush()
        writer.add_image("Prediction", torch.matmul(model.A_mean.unsqueeze(0),model.P_mean.unsqueeze(0)), step)
    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")

#%% Retrieve the inferred parameters
A_scale = pyro.param("A_scale").detach().to('cpu').numpy()
P_scale = pyro.param("P_scale").detach().to('cpu').numpy()
A_mean = pyro.param("A_mean").detach().to('cpu').numpy()
P_mean = pyro.param("P_mean").detach().to('cpu').numpy()

# Print the shapes of the inferred parameters
print("Inferred A shape:", A_mean.shape)
print("Inferred P shape:", P_mean.shape)

#print("Inferred A:", A_mean)
#print("Inferred P:", P_mean)

# Draw A_true and A_shape using imgshow
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(A_true)
plt.title("True A")
plt.colorbar()
plt.savefig("True_A.png")

plt.figure()
plt.imshow(A_mean)
plt.title("Inferred A")
plt.colorbar()
plt.savefig("Inferred_A.png")

# End timer
print("Time taken:")
print(datetime.now() - startTime)

# # %%
# writer.flush()
#!/usr/bin/env -S python3 -u
#%%
import torch
import pyro
import pyro.optim
from cogaps.model import CoGAPSModel
from cogaps.guide import CoGAPSGuide
from cogaps.utils import generate_test_data
import pyro.poutine as poutine
from datetime import datetime

#%% Import tensorboard & setup writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Define the dimensions of the data and the latent space
num_genes = 5000
num_samples = 10000
num_patterns = 50

# Set device
if torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device=torch.device('cpu')

# Generate synthetic test data
D, A_true, P_true = generate_test_data(num_genes, num_samples, num_patterns)

# Move data to device
D = D.to(device)

#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 1000

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.05})

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
model = CoGAPSModel(D, num_patterns, device=device)

#%% Instantiate the guide
guide = CoGAPSGuide(D, num_patterns, device=device)

#%% Define the inference algorithm
svi = pyro.infer.SVI(model=model,
                    guide=guide,
                    optim=optimizer,
                    loss=loss_fn)

#%% Trace
#trace = poutine.trace(model(D)).get_trace()
#trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
#print(trace.format_shapes())

#%% Run inference
for step in range(num_steps):
    loss = svi.step(D)
    if step % 10 == 0:
        writer.add_scalar("Loss/train", loss, step)
        writer.flush()
    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")

#%% Retrieve the inferred parameters
A_scale = pyro.param("A_scale").detach().to('cpu').numpy()
P_scale = pyro.param("P_scale").detach().to('cpu').numpy()

# Print the shapes of the inferred parameters
print("Inferred A shape:", A_scale.shape)
print("Inferred P shape:", P_scale.shape)

# End timer
print("Time taken:")
print(datetime.now() - startTime)

# %%
writer.flush()
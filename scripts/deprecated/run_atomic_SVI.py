#!/usr/bin/env -S python3 -u
#%%
import torch
import pyro
import pyro.optim
from models.model import CoGAPSModel, CoGAPSModelWithAtomicPriorGibbs
from models.guide import CoGAPSGuide, CoGAPSGuideWithAtomicPriorGibbs
from models.utils import generate_test_data
import pyro.poutine as poutine
from datetime import datetime

# Define the dimensions of the data and the latent space
num_genes = 10
num_samples = 5
num_patterns = 2

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
num_steps = 6000

# Setup the atomic prior parameters
lA = 0.1
lP = 0.1
num_atoms_A = 100
num_atoms_P = 100

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.05})

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
model = CoGAPSModelWithAtomicPriorGibbs(D, num_patterns, lA, lP, num_atoms_A, num_atoms_P, device=device)

#%% Instantiate the guide
guide = CoGAPSGuideWithAtomicPriorGibbs(D, num_patterns, lA, lP, num_atoms_A, num_atoms_P, device=device)

#%% Define the inference algorithm
svi = pyro.infer.SVI(model=model.forward,
                    guide=guide.forward,
                    optim=optimizer,
                    loss=loss_fn)

#%% Trace
#trace = poutine.trace(model(D)).get_trace()
#trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
#print(trace.format_shapes())

#%% Run inference
for step in range(num_steps):
    loss = svi.step(D)
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

import torch
import pyro
import pyro.optim
from cogaps.model import cogaps_model
from cogaps.guide import guide
from cogaps.utils import generate_test_data
import pyro.poutine as poutine
from datetime import datetime

# Define the dimensions of the data and the latent space
num_genes = 5000
num_samples = 8000
num_patterns = 50

# Set device
device = 'cpu'

# Generate synthetic test data
D, A_true, P_true = generate_test_data(num_genes, num_samples, num_patterns)

# Move data to device
D = D.to(device)

# Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 1000

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.05})

# Define the inference algorithm
svi = pyro.infer.SVI(model=cogaps_model,
                    guide=guide,
                    optim=optimizer,
                    loss=pyro.infer.Trace_ELBO())

# Trace
trace = poutine.trace(cogaps_model(D,num_patterns)).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())

# Run inference
for step in range(num_steps):
    loss = svi.step(D, num_patterns)
    if step % 100 == 0:
        print(f"Step {step}, ELBO loss: {loss}")

# Retrieve the inferred parameters
A_loc = pyro.param("A_loc").detach().numpy()
P_loc = pyro.param("P_loc").detach().numpy()

# Print the shapes of the inferred parameters
print("Inferred A shape:", A_loc.shape)
print("Inferred P shape:", P_loc.shape)

# End timer
print("Time taken:")
print(datetime.now() - startTime)
#%%
import os
from datetime import datetime
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import pyro
import pyro.optim
import torch
import random
from pyroNMF.run_v2 import *
import random
import time
from tqdm import tqdm
import pickle

#%%
#random.seed(123)

# %%
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes
#data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset
data.obsm['spatial'] = coords.to_numpy() # samples x 2

#%%
# sub sample random incremnts of 1000 cells to check time
# Parameters
#sample_sizes = range(1000, 27000, 1000)  # 1000 to 26000 by increments of 1000
sample_sizes = range(5000, 26000, 5000)
#num_repeats = 5
#sample_sizes = range(5000, 11000, 5000)
num_repeats = 3

results = []

# Main loop
for sample_size in tqdm(sample_sizes, desc="Sample sizes"):
    print(f"\nTesting sample size: {sample_size}")
    
    for repeat in tqdm(range(num_repeats), desc="Repeats", leave=False):
        # Random subsample
        np.random.seed(repeat)  # For reproducibility within each repeat
        indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
        data_subset = data[indices].copy()
        
        # Measure runtime
        start_time = time.time()
        
        try:
            nmf_res = run_nmf_unsupervised(
                data_subset, 
                20, 
                num_steps=10000, 
                spatial=False, 
                plot_dims=[5,4], 
                use_tensorboard_id=f'time_unsupervised_{sample_size}_{repeat}'
            )
            end_time = time.time()
            runtime = end_time - start_time
            success = True
            
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            success = False
            print(f"Error at size {sample_size}, repeat {repeat}: {str(e)}")
        
        # Store results
        results.append({
            'sample_size': sample_size,
            'repeat': repeat,
            'runtime': runtime,
            'success': success,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'adata_runtime': nmf_res.uns['runtime (seconds)']
        })
        nmf_res.write_h5ad(f'/raid/kyla/projects/pyro_NMF/analyses/iterate_samples/time_unsupervised_{sample_size}_{repeat}.h5ad')
        
        # Print progress
        if repeat == 0:  # Print first run of each size
            print(f"  First run: {runtime:.2f} seconds")

# Convert to DataFrame
results_df = pd.DataFrame(results)


# Save results
results_df.to_csv('/raid/kyla/projects/pyro_NMF/analyses/iterate_samples/nmf_runtime_results.csv', index=False)
with open('/raid/kyla/projects/pyro_NMF/analyses/iterate_samples/nmf_runtime_results.pkl', 'wb') as f:
    pickle.dump(results_df, f)

print(f"\nCompleted! Results saved to nmf_runtime_results.csv and .pkl")
print(f"Total successful runs: {results_df['success'].sum()}")
print(f"Total failed runs: {(~results_df['success']).sum()}")




# %%

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Runtime vs Sample Size (all points)
plt.subplot(2, 2, 1)
successful_data = results_df[results_df['success']]
if len(successful_data) > 0:
    plt.scatter(successful_data['sample_size'], successful_data['runtime'], 
               alpha=0.6, s=20)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Sample Size (All Runs)')
    plt.grid(True, alpha=0.3)

# Plot 2: Runtime vs Sample Size (mean with error bars)
plt.subplot(2, 2, 2)
if len(successful_data) > 0:
    summary_stats = successful_data.groupby('sample_size')['runtime'].agg(['mean', 'std'])
    plt.errorbar(summary_stats.index, summary_stats['mean'], 
                yerr=summary_stats['std'], fmt='o-', capsize=5)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (seconds)')
    plt.title('Mean Runtime vs Sample Size (±1 SD)')
    plt.grid(True, alpha=0.3)

# Plot 3: Runtime distribution by sample size (boxplot for selected sizes)
plt.subplot(2, 2, 3)
if len(successful_data) > 0:
    # Select a few sample sizes for boxplot to avoid overcrowding
    selected_sizes = [1000, 5000, 10000, 15000, 20000, 25000]
    selected_sizes = [s for s in selected_sizes if s in successful_data['sample_size'].values]
    
    if selected_sizes:
        box_data = [successful_data[successful_data['sample_size'] == size]['runtime'].values 
                   for size in selected_sizes]
        plt.boxplot(box_data, labels=selected_sizes)
        plt.xlabel('Sample Size')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime Distribution (Selected Sizes)')
        plt.xticks(rotation=45)

# Plot 4: Success rate by sample size
plt.subplot(2, 2, 4)
success_rate = results_df.groupby('sample_size')['success'].mean()
plt.plot(success_rate.index, success_rate.values, 'o-')
plt.xlabel('Sample Size')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Sample Size')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/raid/kyla/projects/pyro_NMF/analyses/iterate_samples/nmf_runtime_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as nmf_runtime_analysis.png")

# %%

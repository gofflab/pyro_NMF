#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
import pandas as pd
import anndata as ad


#%%
def plot_multisample(pattern_i, pattern_name, coords, sampleDf):
    unique_samples = sampleDf.unique()
    num_slices_to_plot = len(unique_samples)

    df_list = []
    for sample in unique_samples:
        sample_ids = sampleDf[sampleDf == sample].index
        sample_coords = coords.loc[sample_ids,:]

        sample_values = pattern_i[sample_ids]

        df = pd.DataFrame({
            "x": sample_coords.iloc[:, 0],
            "y": sample_coords.iloc[:, 1],
            "values": sample_values,
            "sample": sample,
        })

        df_list.append(df)

    df = pd.concat(df_list, axis=0)

    # Get the min and max values for the color bar
    vmin = df["values"].min()
    #vmin = np.percentile(df["values"], 5)
    #vmax = np.percentile(df["values"], 95)
    vmax = df["values"].max()

    g = sns.FacetGrid(df, col="sample", col_wrap=4, height=3)  # Adjust col_wrap for grid layout
    g.fig.suptitle(f'{pattern_name}', fontsize=16)

    # Set consistent axis limits and aspect ratio
    #x_min = min(df['x'])
    #x_max = max(df['x'])

    #y_min = min(df['y'])
    #y_max = max(df['y'])

    for ax, sample in zip(g.axes.flat, unique_samples):
        # Filter data for the current sample
        sample_data = df[df["sample"] == sample]

        # Plot scatter plot on each axis using sns.scatterplot
        scatter = ax.scatter(
            sample_data["x"],
            sample_data["y"],
            c=sample_data["values"],
            s=5, alpha=0.5, cmap="viridis", vmin=vmin, vmax=vmax, edgecolors='none'
        )

        #ax.set_xlim(x_min, x_max)
        #ax.set_ylim(y_min, y_max)

        # Set equal aspect ratio for each axis
        ax.set_aspect('equal', adjustable='box')

        # Remove axis labels and ticks
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label


    # Add color bar using the actual 'values' (no normalization)
    sm = ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # The colorbar doesn't need data array, it's set via the norm
    g.fig.colorbar(sm, ax=g.axes, orientation="vertical", fraction=0.02)
    #g.fig.subplots_adjust(right=.95)

    g.set_titles(col_template="Sample: {col_name}")  # Dynamic titles for each plot

    # Adjust layout and show the plot
    g.tight_layout()
    g.fig.subplots_adjust(right=0.95)  # Adjust space on the right for the color bar
    g.fig.subplots_adjust(top=.95)  # Adjust for the suptitle
    #plt.show()
    return g



def plot_multisample_scale(pattern_i, pattern_name, coords, sampleDf):
    unique_samples = sampleDf.unique()
    num_slices_to_plot = len(unique_samples)

    df_list = []
    for sample in unique_samples:
        sample_ids = sampleDf[sampleDf == sample].index
        sample_coords = coords.loc[sample_ids,:]

        sample_values = pattern_i[sample_ids]

        df = pd.DataFrame({
            "x": sample_coords.iloc[:, 0],
            "y": sample_coords.iloc[:, 1],
            "values": sample_values,
            "sample": sample,
        })

        df_list.append(df)

    df = pd.concat(df_list, axis=0)

    # Get the min and max values for the color bar
    #vmin = df["values"].min()
    vmin = np.percentile(df["values"], 5)
    vmax = np.percentile(df["values"], 95)
    #vmax = df["values"].max()

    g = sns.FacetGrid(df, col="sample", col_wrap=4, height=3)  # Adjust col_wrap for grid layout
    g.fig.suptitle(f'{pattern_name}', fontsize=16)

    # Set consistent axis limits and aspect ratio
    #x_min = min(df['x'])
    #x_max = max(df['x'])

    #y_min = min(df['y'])
    #y_max = max(df['y'])

    for ax, sample in zip(g.axes.flat, unique_samples):
        # Filter data for the current sample
        sample_data = df[df["sample"] == sample]

        # Plot scatter plot on each axis using sns.scatterplot
        scatter = ax.scatter(
            sample_data["x"],
            sample_data["y"],
            c=sample_data["values"],
            alpha = (sample_data["values"]+50)/(df["values"].max()+50),
            #alpha = 0.8,
            s=6, cmap="viridis", vmin=vmin, vmax=vmax, edgecolors='none'
        )

        #ax.set_xlim(x_min, x_max)
        #ax.set_ylim(y_min, y_max)

        # Set equal aspect ratio for each axis
        ax.set_aspect('equal', adjustable='box')

        # Remove axis labels and ticks
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label


    # Add color bar using the actual 'values' (no normalization)
    sm = ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # The colorbar doesn't need data array, it's set via the norm
    g.fig.colorbar(sm, ax=g.axes, orientation="vertical", fraction=0.02)
    #g.fig.subplots_adjust(right=.95)

    g.set_titles(col_template="Sample: {col_name}")  # Dynamic titles for each plot

    # Adjust layout and show the plot
    g.tight_layout()
    g.fig.subplots_adjust(right=0.95)  # Adjust space on the right for the color bar
    g.fig.subplots_adjust(top=.95)  # Adjust for the suptitle
    #plt.show()
    return g




def plot_multiple_genes(patterns, coords, sampleDf, num_rows, num_cols):
    # Determine how many rows and columns are needed for the grid
    #num_genes = len(gene_names)
    #num_cols = 5  # Define the number of columns in the grid (adjust as needed)
    #num_rows = (num_genes // num_cols) + (num_genes % num_cols > 0)  # Calculate rows based on number of genes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    for i, pattern_name in enumerate(patterns.columns):
        pattern_i = patterns.loc[:,pattern_name]
        # Call plot_multisample for each gene
        ax = axes[i]
        # Temporarily use `ax` for each subplot within plot_multisample
        plot_multisample(pattern_i, pattern_name, coords, sampleDf)
        plt.sca(ax)  # Switch to the current axis for the plot
        plt.subplots_adjust(hspace=0.5)

    # Remove any empty subplots
    #for i in range(num_genes, len(axes)):
    #    fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()




def list_plot_multisample(patterns, coords, sampleDf):

    plot_list = []

    for pattern_name in patterns.columns:
        pattern_i = patterns.loc[:,pattern_name]
        g = plot_multisample(pattern_i, pattern_name, coords, sampleDf)
        plot_list.append(g)

    return(plot_list)

# Example usage:
# patterns is a list of DataFrames (one for each gene)
# gene_names is a list of gene names (e.g., ["Gene1", "Gene2", "Gene3", ..., "Gene10"])
# plot_multiple_genes(patterns, coords, sampleDf, gene_names)

#%%
# Test
#data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_wMeta_wAnnotations_wAtlasFULL_sub20_KW.h5ad') # samples x genes
#data = data[data.obs.sort_values('z').index,:]
#patterns = data.obsm['atlas']
#pattern_i = patterns.iloc[:,4]*1
#temp_pattern_i =  pd.DataFrame([max(x,0) for x in np.random.normal(2, 2, patterns.shape[0]).tolist()], index=patterns.index).iloc[:,0]
#coords = data.obs.loc[:,['x', 'y']]
#coords[['y']] = -1*coords[['y']]
#sampleDf = data.obs['brain_section_label_x']
#plot_multisample_scale(pattern_i, 'random', coords, sampleDf)
#out = list_plot_multisample(patterns.iloc[:,:4],coords, sampleDf)


# %%

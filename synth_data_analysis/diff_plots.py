"""
Plotter is a class that plots synthetic data vs real data for visual analysis and comparison.
Datasets should be provided in a form of Pandas dataframes.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import loguru
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf


def doy_to_season(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['dayofyear'] >= 79) & (df['dayofyear'] <= 171),  # Spring
        (df['dayofyear'] >= 172) & (df['dayofyear'] <= 265),  # Summer
        (df['dayofyear'] >= 266) & (df['dayofyear'] <= 354),  # Fall
    ]
    seasons = ['Spring', 'Summer', 'Fall']

    # Use np.select for vectorized mapping
    df['season'] = np.select(conditions, seasons, default='Winter')

    return df


class Plotter:
    def __init__(self, real_df: pd.DataFrame, synth_df: pd.DataFrame):
        self.real_df = real_df
        self.synth_df = synth_df
        self.variables = synth_df.columns

        if real_df.shape[-1] > 20:
            raise Warning("There are a lot of columns in the provided dataset. Some plots may appear wierd. \n"
                          "Consider limiting their number by supplying a list of necessary columns to the methods or "
                          "here in the initialisation.")

    def plot_real_vs_synthetic(self, dir_save: str = './outputs/pics',
                               filename='comparison.png',
                               list_variables: list = None,
                               truncate_data: int = 5000) -> None:
        """
        Draws 4 normal time series plots, each per variable in a dataframe
        """
        os.makedirs(dir_save, exist_ok=True)
        loguru.logger.info(f"Initialized with synth cols: {self.variables}")

        # Plot the stuff
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjusted figure size for better readability
        fig.suptitle('Comparison of Real and Synthetic Data', fontsize=16)  # Add a main title
        if list_variables is not None:
            plotted_variables = list_variables
        else:
            plotted_variables = self.variables
        counter = 0
        for row in axs:
            for col in row:
                if counter < len(plotted_variables):  # Ensure counter doesn't exceed available columns
                    variable = plotted_variables[counter]
                    col.plot(self.real_df.loc[:truncate_data, variable].reset_index(drop=True),
                             label=f'Real {variable}', linewidth=2)
                    col.plot(self.synth_df.loc[:truncate_data, variable].reset_index(drop=True),
                             label=f'Synthetic {variable}', linewidth=2, alpha=0.7, color='orange')
                    col.set_title(f'{variable}', fontsize=14)  # Set a larger title font size
                    col.set_xlabel('Index', fontsize=12)  # Add x-axis label
                    col.set_ylabel('Value', fontsize=12)  # Add y-axis label
                    col.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
                    col.legend(fontsize=10, loc='upper right')  # Add a legend
                    counter += 1
                else:
                    col.axis('off')  # Turn off empty subplots

        # Adjust layout for spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Avoid overlap with the suptitle
        plt.savefig(os.path.join(dir_save, filename))
        loguru.logger.info(f"Saved picture to {dir_save}")
        # plt.clf()

    def plot_correlations_sep(self, data_type: str):
        if data_type == 'synthetic' or data_type == 'synth':
            sns.heatmap(pd.DataFrame(self.synth_df[self.variables].corr()), annot=True, fmt=".2f", vmin=0, vmax=1)
        elif data_type == 'real':
            sns.heatmap(pd.DataFrame(self.real_df[self.variables].corr()), annot=True, fmt=".2f", vmin=0, vmax=1)
        else:
            raise Exception("Check data_type. Available options are: 'synthetic', 'synth', 'real'")
        return None

    def plot_correlations_together(self, filename: str = None, selected_columns: list = None):
        """
        Plots Pearson correlations of real and synthetic sets side by side.

        :param selected_columns: columns to consider in the heatmap
        :param filename: str to save the output
        :return: None
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Two plots side by side
        if selected_columns is None:
            selected_columns = self.variables

        # Synthetic Data Correlation Heatmap
        sns.heatmap(
            pd.DataFrame(self.synth_df[selected_columns].corr()),
            annot=True, fmt=".2f", vmin=0, vmax=1, cbar=True,
            ax=axes[0]
        )
        axes[0].set_title("Synthetic Data Correlation")

        # Real Data Correlation Heatmap
        sns.heatmap(
            pd.DataFrame(self.real_df[selected_columns].corr()),
            annot=True, fmt=".2f", vmin=0, vmax=1, cbar=True,
            ax=axes[1]
        )
        axes[1].set_title("Real Data Correlation")

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a file if filename is provided
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()

    def plot_t_sne(self, dir_save: str = './outputs/pics', filename='tsne_plot.png', n_samples=1000, perplexity=30,
                   cols: list = None) -> None:
        """
        Plot t-SNE visualization for real and synthetic datasets.

        Parameters:
        - dir_save (str): Directory to save the plot.
        - filename (str): Filename to save the plot.
        - n_samples (int): Number of samples to use for t-SNE.
        - perplexity (float): Perplexity parameter for t-SNE.
        """
        os.makedirs(dir_save, exist_ok=True)

        # Sample the data if necessary
        real_sample = self.real_df.sample(n=min(n_samples, len(self.real_df)), random_state=42)
        synth_sample = self.synth_df.sample(n=min(n_samples, len(self.synth_df)), random_state=42)

        # Pick particular cols if provided
        if cols is not None:
            real_sample = real_sample[cols].copy()
            synth_sample = synth_sample[cols].copy()

        # Combine the datasets for t-SNE
        combined_data = pd.concat([real_sample, synth_sample], axis=0)
        labels = np.array([0] * len(real_sample) + [1] * len(synth_sample))  # 0 for real, 1 for synthetic

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(combined_data)

        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], label='Real Data', alpha=0.6)
        plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], label='Synthetic Data', alpha=0.6)
        plt.title('t-SNE Visualization of Real vs. Synthetic Data', fontsize=16)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(dir_save, filename))
        plt.show()

    def plot_hists(self, dir_save: str = './outputs/pics', filename='histograms.png', bins=20,
                   cols: list = None):
        """
        Plot histograms for each variable side by side for real and synthetic datasets.

        Parameters:
        - dir_save (str): Directory to save the plot.
        - filename (str): Filename to save the plot.
        - bins (int): Number of bins for the histogram.
        """
        os.makedirs(dir_save, exist_ok=True)
        if cols is None:
            cols = self.variables

        num_vars = len(cols)
        rows = (num_vars + 1) // 2  # Determine the number of rows for the subplot grid

        fig, axs = plt.subplots(rows, 2, figsize=(20, 5 * rows))  # Adjust figure size dynamically
        fig.suptitle('Histograms of Real and Synthetic Data', fontsize=16)

        axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

        for i, variable in enumerate(cols):
            ax = axs[i]
            # Plot histograms for real and synthetic data
            ax.hist(self.real_df[variable], bins=bins, alpha=0.7, label=f'Real {variable}', color='purple',
                    edgecolor='black')

            ax.hist(self.synth_df[variable], bins=bins, alpha=0.7, label=f'Synthetic {variable}', color='orange',
                    edgecolor='black')

            ax.set_title(f'{variable}', fontsize=14)
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)

        # Turn off unused subplots if any
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(dir_save, filename))
        plt.show()

    def pairplots(self):
        # ToDo
        pass

    def season_plots(self, value_column: str, dir_save: str = '', filename: str = None):
        if 'season' not in self.real_df.columns or 'season' not in self.synth_df.columns:
            self.real_df = doy_to_season(self.real_df)
            self.synth_df = doy_to_season(self.synth_df)

        if value_column not in self.real_df.columns or value_column not in self.synth_df.columns:
            raise ValueError(f"Both DataFrames must contain the '{value_column}' column.")

        avg_values = self.real_df.groupby('season')[value_column].mean()
        synth_avg_values = self.synth_df.groupby('season')[value_column].mean()

        self.real_df.drop(columns='season', inplace=True)
        self.synth_df.drop(columns='season', inplace=True)

        seasons = avg_values.index
        x = range(len(seasons))

        plt.figure(figsize=(8, 5))
        plt.bar(x, avg_values.values, width=0.4, label='Real', color='purple', alpha=0.7, edgecolor='black',
                align='center')
        plt.bar([i + 0.4 for i in x], synth_avg_values.values, width=0.4, label='Synthetic', alpha=0.7, color='orange',
                edgecolor='black', align='center')

        plt.xlabel('Season')
        plt.ylabel(f'Average {value_column}')
        plt.title(f'Average {value_column} by Season')
        plt.xticks([i + 0.2 for i in x], seasons, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_autocorrelations(self, selected_columns: list, filename: str=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, selected_column in enumerate(selected_columns[:4]):
            plot_acf(self.real_df[selected_column],
                     title=f'Real {selected_column}',
                     lags=100,
                     ax=axes[0, i])

            plot_acf(self.synth_df[selected_column],
                     title=f'Generated {selected_column}',
                     lags=100,
                     ax=axes[1, i])

        if filename is not None:
            plt.savefig(filename)
import pandas as pd
import matplotlib.pyplot as plt


# Set global font sizes
plt.rcParams.update({
    'font.size': 18,          # Default font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 18,     # Axis label font size
    'xtick.labelsize': 18,    # X tick font size
    'ytick.labelsize': 18,    # Y tick font size
    'legend.fontsize': 18,    # Legend font size
    'figure.titlesize': 20    # Figure title font size (if used)
})

# Load data
df = pd.read_csv('exp_pred/eva_results.csv', index_col=0)

# Convert Index to percentage
df['Index'] = df['Index'].astype(float) * 100

# Unique models
models = df.index.unique()

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Colors and styles
styles = {
    'DoppelGANger': ('--', 'o'),
    'copula': (':', 's'),
    'flow': ('-', 'o'),
    'gmm': (':', 'D'),
    'real': ('-', 'o')  # If you later add real data
}

# First subplot: MAE
for model in models:
    model_df = df.loc[model]
    linestyle, marker = styles.get(model, ('-', 'o'))
    axs[0].plot(model_df['Index'], model_df['MAE'], linestyle=linestyle, marker=marker, label=model)
axs[0].set_title('Forecast Horizon: 24h - MAE')
axs[0].set_ylabel('MAE')
axs[0].legend(title='Model')
axs[0].grid(True)

# Second subplot: RMSE
for model in models:
    model_df = df.loc[model]
    linestyle, marker = styles.get(model, ('-', 'o'))
    axs[1].plot(model_df['Index'], model_df['RMSE'], linestyle=linestyle, marker=marker, label=model)
axs[1].set_title('Forecast Horizon: 24h - RMSE')
axs[1].set_xlabel('% of real dataset used for training')
axs[1].set_ylabel('RMSE')
axs[1].legend(title='Model')
axs[1].grid(True)

plt.tight_layout()
plt.show()
# Save the figure
fig.savefig('exp_pred/plot_results.png', dpi=300)
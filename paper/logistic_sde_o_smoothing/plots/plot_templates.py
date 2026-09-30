import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set defaults for plot labels and titles
import matplotlib as mpl

mpl.rcParams.update({
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Change default styles
sns.set_style("darkgrid")
sns.set_palette("Set2")

# ---------------------------------------- Basic single lineplot/scatterplot ----------------------------------------------

# DELETE LINE/SCATTER COMMANDS AS APPROPRIATE
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# np arrays for the scatter points
X = np.random.uniform(low=-1., high=1., size=(100,))
y = np.sin(np.pi * X) + np.random.normal(loc=0., scale=0.1, size=(100,)) 

# np arrays for the line
x_true = np.linspace(start=-1., stop=1., num=1001)
y_true = np.sin(np.pi * x_true)

# Line plot command
line_2d_kwargs = dict(lw=2.0, ls='-', marker=None, ms=None, alpha=1.0) #mfc='blue', mec='black', 
ax.plot(x_true, y_true, color="#d71212", label=r'$f(x)=\sin(\pi x)$', **line_2d_kwargs)

# Scatter plot command
scatter_kwargs = dict(s=36, marker='o', alpha=1.0, edgecolors='black', linewidths=0.8)
ax.scatter(X, y, c="#1dc931", label=r'$\mathcal{D}$', **scatter_kwargs)

# Everything else
ax.set(xlabel=r'$x$', ylabel=r'$y$', xlim=(-1., 1.), ylim=(-1.3, 1.3), title='')
ax.tick_params(axis='both', length=6, pad=4, labelsize=10)
ax.grid()
ax.legend()

plt.show()

# ---------------------------------------- Histogram, via mpl --------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sample = np.random.normal(loc=0., scale=1., size=1000)

# Main call
hist_kwargs = dict(bins=30, histtype='bar', density=True, stacked=False)
patch_kwargs = dict(facecolor='blue', edgecolor='black', alpha=0.5, label=r'\mathcal{N}(0, 1)') #mfc='blue', mec='black', 
ax.hist(sample, **hist_kwargs, **patch_kwargs)

ax.tick_params(axis='both', length=6, pad=4, labelsize=10)


# ---------------------------------------- Boxplot via sns --------------------------------------------------

titanic_df = sns.load_dataset('titanic')

# sns Boxplot/Catplot Axis Level argument example
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Formatting passed to sns.boxplot - the ax.boxplot kwargs will override these ones
sns_kwargs = dict(palette={"male": "#2cc91d", "female": "#e21f0a"}, whis=(5., 95.), width=0.8, fill=True)

# Key arguments to the call only
sns.boxplot(data=titanic_df, ax=ax, x='age', y='deck', hue='sex', **sns_kwargs)

ax.grid()
ax.tick_params(axis='both', length=6, pad=4, labelsize=10)

plt.show()

# -----------------------------------------------------------------------------------------------------------------

# Full formatting options that can be passed to ax.boxplot - can be used to override the sns.boxplot kwargs
medianprops=dict(lw=2.0, color='black', ls='-')
flierprops=dict(marker='x', markersize=5.0, color='black')
boxprops=dict(facecolor='blue', alpha=0.8, edgecolor='black', lw=1.5)
capprops=dict(lw=1.0, color='black', ls='-')
whiskerprops=dict(lw=1.0, color='black', ls='-')
props_kwargs = dict(whiskerprops=whiskerprops, capprops=capprops, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops)

# ------------------------------------------- Custom errorbar plot function-------------------------------------------------------------

def ebar_plot(ax: matplotlib.axes.Axes, data: pd.DataFrame, x: str, y: str, hue: str, **ebar_kwargs):
    """
    Errorbar plot. Similar in spirit to sns.pointplot, but with the 'categorical' variable
    instead treated as a float, allowing linear/logarithmic scaling of the axis representing
    the category.
    
    Inputs
    --------------
    ax (matplotlib.axes.Axes): Axis for the plot
    data (pd.DataFrame): The data
    x (str): Column name of the data that represents the category.
    y (str): Column name of the data that represents the metric of interest.
    hue (str): Column name of the data that represents the treatment.

   Returns
    --------------
    ax (matplotlib.axes.Axes): Axis for the plot    
    """
    group_df = data.groupby([hue, x])[y].agg(['mean', 'min', 'max'])
    group_df = group_df.reset_index()
    hue_names = list(group_df[hue].unique())
    for name in hue_names:
        group_hue_name_df = group_df[group_df[hue] == name].sort_values(by=x)
        x_vals = group_hue_name_df[x]
        mean = group_hue_name_df['mean'].to_numpy()
        _max = group_df[group_df[hue] == name]['max'].to_numpy()
        _min = group_df[group_df[hue] == name]['min'].to_numpy()
        yerr = np.abs(np.stack([_min, _max], axis=0) - mean)
        ax.errorbar(x_vals, mean, yerr=yerr, label=name, **ebar_kwargs)
    ax.set(xlabel=x, ylabel=y)
    ax.legend()
    return ax
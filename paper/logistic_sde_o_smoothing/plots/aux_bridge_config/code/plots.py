import numpy as np
import pandas as pd
import json

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from particles_cdssm.tools import struct_arrs_to_arr, build_cdssm
from particles_cdssm.cdssm_lib import CDSSM_LIB


# ------------------LOAD THE OUTPUTS--------------------

def load_config(config_name: str) -> dict:
    config_path = "../../../config/" + config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def build_errors_df(smc_df: pd.DataFrame, ground_truth_df: pd.DataFrame, config: dict):
    ground_truth_df = ground_truth_df.set_index(['cdssm', 'add_func'])
    bias_df = smc_df.copy()
    cdssm_names = [c for c in config["cdssm"].keys()]
    add_funcs = config["add_funcs"]
    Ts = [str(t) for t in config["Ts"]]
    for t in Ts:
        for cdssm_name in cdssm_names:
            for add_func_name in add_funcs:
                mask = (smc_df['cdssm'] == cdssm_name) & (smc_df['add_func'] == add_func_name)
                bias_df.loc[mask, t] = smc_df.loc[mask, t] - ground_truth_df.at[(cdssm_name, add_func_name), t]
    return bias_df

def wide_to_long(df: pd.DataFrame, config: dict):
    str_Ts = [str(t) for t in config['Ts']]
    id_vars = [c for c in df.columns if c not in str_Ts]
    long_df = pd.melt(df, id_vars = id_vars, value_vars=str_Ts, var_name='t', value_name='estimate')
    long_df['t'] = long_df['t'].astype(np.int64)
    return long_df

def load(config_name: str, remote=False) -> dict:
    """
    Load the output dataframes from the numerical experiment, preprocess the outputs
    """
    local_or_remote = 'remote' if remote else 'local'
    data = {}
    data['config'] = load_config(config_name)
    data['smc_wide'] = pd.read_json(f'../../../results/smc/{local_or_remote}/r_' + config_name)
    data['smc_long'] = wide_to_long(data['smc_wide'], data['config'])
    if ('run_lgssm' in data['config'] and data['config']['run_lgssm']) or ('lgssm' in data['config'] and data['config']['lgssm']):
        data['ground_truth'] = pd.read_json(f'../../../results/ground_truth/{local_or_remote}/r_' + config_name)
        data['errors_wide'] = build_errors_df(data['smc_wide'], data['ground_truth'], data['config'])
        data['errors_long'] = wide_to_long(data['errors_wide'], data['config'])
    return data

# ------------------PLOT THE RESULTS--------------------

def logistic_cdssm(lamperti=True):
    """
    Plots the Logistic CD-SSM and its Lamperti transform.
    """
    T = 10
    N_IMPUTED = 100
    THETA_3 = 0.7

    def lamperti_transform(x: np.ndarray, theta_3: float = THETA_3) -> np.ndarray:
        """Apply X = log(P) / theta_3."""
        x = np.asarray(x, dtype=np.float64)
        if np.any(x <= 0):
            raise ValueError("The Lamperti transform requires strictly positive values.")
        return np.log(x) / theta_3

    def inverse_lamperti_transform(x: np.ndarray, theta_3: float = THETA_3) -> np.ndarray:
        """Apply P = exp(theta_3 * X)."""
        x = np.asarray(x, dtype=np.float64)
        return np.exp(theta_3 * x)

    def plot_cdssm(
        ax: matplotlib.axes.Axes,
        x_t: np.ndarray,
        y_t: np.ndarray,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
    ):
        ax.plot(
            x_t[:len(x_arr)],
            x_arr,
            color="#23a437",
            lw=2.0,
            label="Logistic SDE path",
        )

        ax.scatter(
            y_t[:len(y_arr)],
            y_arr,
            color="#010101",
            marker="x",
            s=48,
            linewidths=1.5,
            zorder=10,
            label="Observations",
        )

        ax.set_title("Logistic growth diffusion with negative-binomial observations")
        ax.set_ylabel("Population size")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_xlim(0, T)

        return ax

    def plot_lamperti_transformed_sde(
        ax: matplotlib.axes.Axes,
        x_t: np.ndarray,
        x_arr: np.ndarray,
    ):
        lamperti_arr = lamperti_transform(x_arr)

        ax.plot(
            x_t[:len(lamperti_arr)],
            lamperti_arr,
            color="#e30808",
            lw=2.0,
            label=r"Lamperti path: $\log(P_t)/\theta_3$",
        )

        ax.set_title("Lamperti-transformed logistic diffusion")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$X_t = \log(P_t) / \theta_3$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_xlim(0, T)

        return ax

    def produce_figure(x_t: np.ndarray, y_t: np.ndarray, x_arr: np.ndarray, y_arr: np.ndarray):
        fig, axes = plt.subplots(2,1, figsize=(16, 6), sharex=True, gridspec_kw={"hspace": 0.45})
        
        axes[0] = plot_cdssm(axes[0], x_t, y_t, x_arr, y_arr)
        axes[1] = plot_lamperti_transformed_sde(axes[1], x_t, x_arr)
        
        fig.subplots_adjust(left=0.07, right=0.98)
        
        return fig, axes
            
        
    cdssm_name = 'LAMPERTI_LOGISTIC_NB_LOW' if lamperti else 'LOGISTIC_NB_LOW'
    cdssm = build_cdssm(CDSSM_LIB[cdssm_name])
    
    np.random.seed(55765)
    x, y = cdssm.simulate(T, num=N_IMPUTED)

    x_arr = struct_arrs_to_arr(x).ravel()
    y_arr = np.array(y).ravel()

    if lamperti:
        x_arr = inverse_lamperti_transform(x_arr, theta_3=THETA_3)
    
    x_t = np.linspace(start=1/len(x_arr), stop=T, num=len(x_arr))
    y_t = np.arange(1, T+1, dtype=np.int64)

    fig, axes = produce_figure(x_t, y_t, x_arr, y_arr)
    return fig, axes

def cpu_times_boxplot(ax, data: dict):
    
    smc_df = data['smc_wide']
    
    sns.boxplot(
        ax=ax,
        data=smc_df,
        x='algorithm',
        y='cpu',
        color='forestgreen',     # internal box colour
        width=0.55,
        linewidth=2.0,        # thicker box/whisker lines
        fliersize=6,
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'white',
            'markeredgecolor': '0.25',
            'markeredgewidth': 1.4
        }
    )

    ax.set_xticklabels([
        "GT: $\\mathcal{O}(N)$\n$N=44100$",
        "FFBS: $\\mathcal{O}(N^2)$\n$N=100$",
        "FFBS-MCMC: $\\mathcal{O}(N)$\n$N=11000$"
    ])

    ax.set_xlabel("")
    ax.set_ylabel("CPU time", fontsize=12)

    ax.tick_params(axis='both', labelsize=11)
    ax.tick_params(axis='x', pad=8)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25)
    ax.xaxis.grid(False)

    sns.despine(ax=ax)
    return ax

def estr_single_boxplot(ax, data, t):
    """
    Plot the boxplot of the estimators for a single value of t.
    
    Axis level function
    """

    XLABELS = [
        "GT: $\\mathcal{O}(N)$\n$N=44100$",
        "FFBS-$\\mathcal{O}(N^2)$\n$N=100$",
        "FFBS-MCMC: $\\mathcal{O}(N)$\n$N=11000$"
    ]

    smc_df = data['smc_wide']
    
    sns.boxplot(
        ax=ax,
        data=smc_df,
        x='algorithm',
        y=str(t),
        width=0.55,
        linewidth=2.0,
        color='steelblue',
        boxprops={'edgecolor': '0.2', 'linewidth': 2},
        whiskerprops={'color': '0.2', 'linewidth': 2},
        capprops={'color': '0.2', 'linewidth': 2},
        medianprops={'color': '0.15', 'linewidth': 2.5},
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'white',
            'markeredgecolor': '0.25',
            'markeredgewidth': 1.4,
            'markersize': 6
        }
    )

    # Labels
    ax.set_xticklabels(XLABELS)
    ax.set_xlabel("")
    ax.set_ylabel(r'$\hat{\mathbb{Q}}_t^N(\Phi_t)$', fontsize=13)
    ax.set_title(fr"$t={t}$", fontsize=13)

    # Ticks
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='x', pad=8)

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25)
    ax.xaxis.grid(False)

    sns.despine(ax=ax)

def estr_boxplots(data, t_values=('1', '10', '100', '1000', '10000')):
    """Plot boxplots for multiple values of t.
    
        Figure level function
    """

    fig, axes = plt.subplots(
        1,
        len(t_values),
        figsize=(4 * len(t_values), 4),
        sharey=False
    )

    for ax, t in zip(axes, t_values):
        estr_single_boxplot(ax, data, t)

    # Only show the y-axis label on the first panel
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.tight_layout()

    return fig, axes

def estr_lineplot(ax, data): 
    
    smc_long = data['smc_long']
    
    sns.lineplot(
        ax=ax,
        data=smc_long,
        x='t',
        y='estimate',
        hue='algorithm',
        estimator=lambda x: np.var(x, ddof=1),
        errorbar=("ci", 95),
        n_boot=1000,
        linewidth=2.0
    )

    # Log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Axis labels
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\hat{\mathbb{Q}}_t^N(\Phi_t)$', fontsize=13)

    # Replace legend labels
    handles, _ = ax.get_legend_handles_labels()

    legend_labels = [
        r'GT: $\mathcal{O}(N)$, $N=44100$',
        r'FFBS-$\mathcal{O}(N^2)$, $N=100$',
        r'FFBS-MCMC: $\mathcal{O}(N)$, $N=11000$'
    ]

    ax.legend(
        handles=handles,
        labels=legend_labels,
        frameon=False,
        fontsize=10,
        loc='upper left'
    )

    # Cosmetic improvements
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', length=2)

    ax.set_axisbelow(True)
    ax.grid(
        True,
        which='major',
        linestyle='--',
        linewidth=0.8,
        alpha=0.3
    )

    # Avoid dense minor grid lines on the log axes
    ax.grid(False, which='minor')
    ax.set_xlim((1, 10**4))
    sns.despine(ax=ax)
    return ax
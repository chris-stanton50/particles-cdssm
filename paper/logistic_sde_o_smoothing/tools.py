import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from particles_cdssm.cdssm_lib import CDSSM_LIB
from particles_cdssm.tools import struct_arrs_to_arr
from particles_cdssm.tools import build_cdssm

def load_config(config_name: str) -> dict:
    config_path = "./config/" + config_name
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

def load(config_name: str) -> dict:
    """
    Load the output dataframes from the numerical experiment, preprocess the outputs
    """
    data = {}
    data['config'] = load_config(config_name)
    data['smc_wide'] = pd.read_json('./results/smc/r_' + config_name)
    data['smc_long'] = wide_to_long(data['smc_wide'], data['config'])
    if 'run_lgssm' in data['config'] and data['config']['run_lgssm'] or 'lgssm' in data['config'] and data['config']['lgssm']:
        data['ground_truth'] = pd.read_json('./results/ground_truth/r_' + config_name)
        data['errors_wide'] = build_errors_df(data['smc_wide'], data['ground_truth'], data['config'])
        data['errors_long'] = wide_to_long(data['errors_wide'], data['config'])
    return data

# --------- Aggregations performed with 'groupby': for use in plots with matplotlib ---------

def variance(Ts: list, smc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metric used to measure performance in Poyiadjis et al (2011)
    """
    str_Ts = [str(t) for t in Ts]
    smc_df_f = smc_df.loc[:, ['add_func', 'algorithm', 'fk'] + str_Ts]
    var_df = smc_df_f.groupby(['add_func', 'algorithm', 'fk'])[str_Ts].var()
    return var_df

def iqr(Ts: list, smc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metric used to measure performance in Chopin and Papasiliopoulos (2020)
    """
    str_Ts = [str(t) for t in Ts]
    smc_df_f = smc_df.loc[:, ['add_func', 'algorithm', 'fk'] + str_Ts]
    upper_q_df = smc_df_f.groupby(['add_func', 'algorithm', 'fk'])[str_Ts].quantile(0.75)
    lower_q_df = smc_df_f.groupby(['add_func', 'algorithm', 'fk'])[str_Ts].quantile(0.25)
    iqr_df = upper_q_df - lower_q_df
    return iqr_df

def sq_iqr(Ts: list, smc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metric used to measure performance in Dau and Chopin (2023)
    """
    iqr_df = iqr(Ts, smc_df)
    sq_iqr_df = iqr_df ** 2
    return sq_iqr_df

def bias(Ts: list, smc_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    iqr_df = iqr(Ts, smc_df)
    sq_iqr_df = iqr_df ** 2
    return sq_iqr_df

# --------- Functions to produce the plots ---------

"""
Plot types for each config:

- Plot of a simulation from the CD-SSM
- Boxplots of the CPU times for a given FK and add func
- Boxplots of the
- Lineplots of the (x) of the smc estimators of the additive functional:
    - x = Variance/IQR/IQR^2/Bias^2/MSE/MAE

"""

def plot_ou_cdssm():
    T=20; num_simulation=1000
    cdssm = build_cdssm(CDSSM_LIB['OU_0.2'])
    s_ts = cdssm.s_ts

    np.random.seed(55765)
    x, y = cdssm.simulate(T=20, num=num_simulation)

    x_t = np.linspace(start=0, stop=T, num=(T*num_simulation)+1)[1:]
    y_t = np.arange(1, 21, dtype=np.float64)

    x_arr = struct_arrs_to_arr(x).ravel()
    y_arr = np.array(y).ravel()

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Line plot command
    line_kwargs = dict(lw=2.0, ls="-", marker=None, alpha=1.0)

    line_2d_kwargs = dict(lw=2.0, ls='-', marker=None, ms=None, alpha=1.0) #mfc='blue', mec='black', 
    ax.plot(x_t, x_arr, color="#23a437", **line_2d_kwargs)

    # Scatter plot command
    observation_kwargs = dict(marker="x", s=48, linewidths=1.5, alpha=1.0, zorder=10)
    ax.scatter(y_t, y_arr, c="#010101", **observation_kwargs)

    # Everything else
    x_pad=0.02; y_pad=0.05
    ax.set(xlabel='Time (s)', ylabel=r'$X(s)$', xlim=(-T*s_ts*x_pad, T*s_ts*(1.+x_pad)), ylim=(np.min(x_arr)*(1.+y_pad), np.max(x_arr)*(1.+y_pad)), title="OU CD-SSM Model Simulation")
    ax.tick_params(axis='both', length=6, pad=4, labelsize=10)
    ax.set_xticks(np.arange(21).tolist())
    ax.grid()
    ax.legend()
    return fig, ax

def plot_logistic_cdssm(lamperti=True):
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
            
        
    cdssm_name = 'LAMPERTI_LOGISTIC_NB_V_HIGH' if lamperti else 'LOGISTIC_NB_V_HIGH'
    cdssm = build_cdssm(CDSSM_LIB[cdssm_name])
    
    # np.random.seed(346345)
    x, y = cdssm.simulate(T, num=N_IMPUTED)

    x_arr = struct_arrs_to_arr(x).ravel()
    y_arr = np.array(y).ravel()

    if lamperti:
        x_arr = inverse_lamperti_transform(x_arr, theta_3=THETA_3)
    
    x_t = np.linspace(start=1/len(x_arr), stop=T, num=len(x_arr))
    y_t = np.arange(1, T+1, dtype=np.int64)

    fig, axes = produce_figure(x_t, y_t, x_arr, y_arr)
    return fig, axes

def estimator_dict():
    estimators = {'variance': 
                            {'df': 'smc_long', 
                            'estimator': lambda x: np.var(x, ddof=1),
                            'ylabel': 'Variance: ' + r'$Var[\hat{\mathbb{Q}}_t^N(\phi_t)]$',
                            },
                    'IQR': 
                            {'df': 'smc_long', 
                            'estimator': lambda x: np.percentile(x, 0.75) - np.percentile(x, 0.25),
                            'ylabel': 'Interquartile Range',
                            },
                    'IQR2': 
                            {'df': 'smc_long', 
                            'estimator': lambda x: (np.percentile(x, 0.75) - np.percentile(x, 0.25)) ** 2,
                            'ylabel': 'Squared Interquartile Range',
                            },
                    'bias': {'df': 'errors_long',
                            'estimator': np.mean,
                            'ylabel': 'Bias ' + r'$E[\hat{\mathbb{Q}}_t^N(\phi_t)] - \mathbb{Q}_t(\phi_t)$',
                            },
                    'bias2': {'df': 'errors_long',
                            'estimator': lambda x: np.mean(x) ** 2,
                            'ylabel': 'Bias^2' + r'($E[\hat{\mathbb{Q}}_t^N(\phi_t)] - \mathbb{Q}_t(\phi_t))^2$',
                            },
                    'mse': {'df': 'errors_long',
                            'estimator': lambda x: np.mean(np.square(x)),
                            'ylabel': 'MSE ' + r'$E[(\hat{\mathbb{Q}}_t^N(\phi_t) - \mathbb{Q}_t(\phi_t))^2]$',
                            },
                    'mae': {'df': 'errors_long',
                            'estimator': lambda x: np.mean(np.abs(x)),
                            'ylabel': 'MAE ' + r'$E[|\hat{\mathbb{Q}}_t^N(\phi_t) - \mathbb{Q}_t(\phi_t)|]$',
                            }
                    }
    return estimators

def algorithms_map(config: dict):
    algs_dict = config['algorithms']
    algos_names_map = {'naive': 'GT', 'ON2': 'FFBS: ' + r'$\mathcal{O}(N^2)$', 'MCMC_1': 'FFBS-MCMC: ' + r'$\mathcal{O}(N)$'}
    algorithms_map = {algo: algos_names_map[algo] + f' \n(N={alg_dict["N"][0]})' for algo, alg_dict in algs_dict.items() if alg_dict['N']}
    return algorithms_map

def cpu_boxplot(ax: matplotlib.axes.Axes, data: dict, cdssm: str, fk_name: str, algorithms=None):
    
    config = data['config']
    smc_df = data['smc_wide']
    algos_map = algorithms_map(config)
    mask = (smc_df['cdssm'] == cdssm) & (smc_df['fk'] == fk_name)
    df = smc_df.loc[mask, :]

    if algorithms:
        mask & smc_df['algorithms'].isin(algorithms)
        algos_map  = {k: v for k, v in algos_map.items() if k in algorithms}
        
    df['algorithm'] = df['algorithm'].map(algos_map)
    sns.boxplot(ax=ax, data=df, x='algorithm', y='cpu', hue='add_func')
    ax.set_title(f'cdssm: {cdssm}, fk_model: {fk_name}')
    return ax

def estr_boxplot(ax, data: dict, cdssm: str, add_func: str, t: int):
    """
    BootstrapDA, FFBS, and that's it! 
    """
    smc_df = data['smc_wide']
    mask = (smc_df['cdssm'] == cdssm) & (smc_df['add_func'] == add_func)
    smc_df_f = smc_df[mask][['algorithm', 'fk', 'N', str(t)]]
    smc_df_f['fk'] = smc_df['fk'].replace({'BsR_DH': 'Bootstrap', 'BootstrapDA': 'Bootstrap'})
    ax = sns.boxplot(data=smc_df_f, x='algorithm', y=str(t), hue='fk', ax=ax)
    if ('run_lgssm' in data['config'] and data['config']['run_lgssm']) or ('lgssm' in data['config'] and data['config']['lgssm']):
        ground_truth_df = data['ground_truth']
        mask = (ground_truth_df['cdssm'] == cdssm) & (ground_truth_df['add_func'] == add_func)
        ground_truth = ground_truth_df[mask][str(t)].iat[0]
        line_2d_kwargs = dict(lw=2.0, ls='--', alpha=0.6, color='black', label='true') 
        x_min, x_max = ax.get_xlim()
        ax.hlines(y=ground_truth, xmin=x_min, xmax=x_max, **line_2d_kwargs)
        ax.set(xlim=(x_min, x_max), xlabel='', ylabel='')
    return ax

def ou_estr_boxplots():
    data= load('config_1.json')
    Ts = [1, 10, 100, 1000, 10000]
    add_funcs = ['FirstMom', 'SecondMom']
    add_funcs_labels_map = {'FirstMom': r'$E[\sum_1^t X(s_t)|Y_{1:t} = y_{1:t}]$', 'SecondMom': r'$E[[\sum_1^t X(s_t)^2|Y_{1:t} = y_{1:t}]$'}

    fig, axes = plt.subplots(len(add_funcs), len(Ts), figsize=(20, 10), sharex=True)

    for i, add_func in enumerate(add_funcs):
        for j, t in enumerate(Ts):
            ax = axes[i,j]
            ax = estr_boxplot(ax, data, 'OU_0.2', add_func, t)
            ax.get_legend().remove()
            if j == 0:
                ax.set_ylabel(add_funcs_labels_map[add_func])
            if i == 0:
                ax.set_title(f't={str(t)}')

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.subplots_adjust(hspace=0.05, bottom=0.07)
    return fig, axes    

def lineplot(ax: matplotlib.axes.Axes, data: dict, estimator: str, cdssm_name: str, add_func_name: str, fk_name: str, T: int, algorithms=None): 
    estimator_d = estimator_dict()[estimator]
    df = data[estimator_d['df']]
    
    mask = (df['cdssm'] == cdssm_name) & (df['add_func'] == add_func_name) & (df['t'] <= T) & (df['fk'] == fk_name)
    if algorithms:
        mask = mask & df['algorithm'].isin(algorithms)
    df = df.loc[mask, :]
        
    sns.lineplot(ax=ax, data=df, x='t', y='estimate', hue='algorithm', estimator=estimator_d['estimator'], errorbar=("ci", 95), n_boot=1000)
    
    ax.set_xticks(np.linspace(start=0, stop=T, num=11).tolist())
    ax.set(xlim=(0, T), ylabel=estimator_d['ylabel'])
    ax.set_title(f'CDSSM: {cdssm_name}, Add Func: {add_func_name}, FK Model: {fk_name}')
    return ax
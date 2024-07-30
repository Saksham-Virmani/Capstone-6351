import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

def data_prep(price, ma=5):
    df = pd.DataFrame(price)
    df['ma'] = price.rolling(ma).mean()
    df['log_returns'] = np.log(price/price.shift(-1)).dropna()
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def initialize_model(model, params):
    for parameter, value in params.items():
        setattr(model, parameter, value)
    return model

def get_regimes_clustering(params):
    clustering = initialize_model(AgglomerativeClustering(), params)
    return clustering

def plot_regimes(plot_data, states, title):
    plt.figure(figsize=(10, 5))
    regime_labels = ['Bear', 'Bull']
    regime_colors = ['red', 'green']
    current_regime = None
    for i, regime in enumerate(states):
        if current_regime is None:
            current_regime = regime
            x = [plot_data.index[0]]
            y = [plot_data.iloc[0]]
        elif current_regime != regime:
            plt.plot(x, y, color=regime_colors[current_regime], label=regime_labels[current_regime])
            current_regime = regime
            x = [plot_data.index[i], plot_data.index[i+1]]
            y = [plot_data.iloc[i], plot_data.iloc[i+1]]
        else:
            x.append(plot_data.index[i+1])
            y.append(plot_data.iloc[i+1])
    if current_regime is not None:
        plt.plot(x, y, color=regime_colors[current_regime])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
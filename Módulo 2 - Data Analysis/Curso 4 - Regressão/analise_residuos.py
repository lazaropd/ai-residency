import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def calc_rss(residuo):
    return float(((residuo) ** 2).sum())
    
def calc_r2(y, y_hat):
    return r2_score(y_hat, y)

def analise_residuos(y, y_hat, graph=False):
    """sendo conhecidos y (pandas column) e y_hat (numpy array Nx1)"""

    size = len(y_hat)
    if isinstance(y, pd.DataFrame): y = np.array(y.values.ravel())
    y_hat = np.array(y_hat)
    res = y - y_hat
    obs = np.arange(1, size+1)

    # mostra resumo estatístico do desempenho do modelo
    print('***********************************************')
    print('Número de observações: ', size)
    print('RSS: %.2f'%calc_rss(res))
    print('R2: %.2f'%calc_r2(y, y_hat))
    print('***********************************************\n')

    if graph:

        # configuração do gráfico para análise de resíduos
        fig, ax = plt.subplots(2, 2, figsize=(16,8))
        fig.suptitle('Análise de Resíduos', fontsize=20)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # análise do resíduo pela ordem das observações, avaliar se o resíduo tem comportamento estável
        ax[0][0].scatter(obs, res, marker='o', c= 'r', alpha=0.8, edgecolors='none')
        ax[0][0].plot(obs, res, c= 'k', lw=0.5, alpha=0.8)
        ax[0][0].plot([0, size], [0, 0], c='k')
        ax[0][0].set_title('Resíduos', size=16)

        # teste de normalidade do resíduo, um modelo bem ajustado deve ter um resíduo totalmente estocástico e normal
        a, result = stats.probplot(res, plot=ax[0][1], dist='norm')
        # teste estatístico se o resíduo segue uma distribuição normal
        _, p = stats.normaltest(res)
        ax[0][1].text(-2, 0.8*res.max(), 'r=%.2f\np-value=%.4f'%(result[2], p))
        ax[0][1].set_title('Normalidade (pp-plot)', size=16)

        # verificação se a premissa de homoscedicidade está sendo atendida (variância do resíduo constante para todo o domínio)
        ax[1][0].scatter(y_hat, res, marker='o', c= 'r', alpha=0.8, edgecolors='none')
        ax[1][0].plot([0, y_hat.max()], [0, 0], c='k')
        ax[1][0].set_title('Resíduos vs Ajustado', size=16)

        # distribuição dos resíduos, aferição visual, verificar se os resíduos formam uma normal (gaussiana)
        ax[1][1].hist(res, density=True, facecolor='b', alpha=0.5, edgecolor='gray')
        rv = stats.norm(res.mean(), res.std())
        x = np.linspace(res.min(), res.max(), 100) 
        h = plt.plot(x, rv.pdf(x), c='b', lw=2)
        ax[1][1].set_title('Histograma', size=16)

        plt.show()
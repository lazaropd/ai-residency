import seaborn as sns
import matplotlib.pyplot as plt

def analise_exploratoria(df, y_cols, ig_cols, graph=False):

    X_cols = [col for col in df.columns if col not in y_cols and col not in ig_cols]

    X = df.loc[:, X_cols]
    y = df.loc[:, y_cols]
    M = df.loc[:, X_cols + y_cols]
    size = len(y)
    print('***********************************************')
    print('Dados extraídos com sucesso')
    print('X: ', ' | '.join(X_cols))
    print('y: ', ' | '.join(y_cols))
    print('M: ', (X.shape), '|', y.shape )
    print('***********************************************')

    if graph:   

        # configuração do gráfico para análise de resíduos
        sns.pairplot(M, height=2)
        plt.show()

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(M.corr(), ax=ax, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')
        plt.show()

    return X, y, M
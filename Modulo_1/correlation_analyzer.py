import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from tabulate import tabulate

class CorrelationAnalyzer:
    """
    Classe para análise de correlação em DataFrames numéricos.
    Oferece funcionalidades de cálculo e visualização.
    """
    
    def __init__(self, data):
        """
        Inicializa o analisador com um DataFrame.
        
        Parâmetros:
        -----------
        data : pd.DataFrame
            Conjunto de dados contendo variáveis numéricas.
        """
        self.data = data.select_dtypes(include='number')
    
    def correlation(self, target_col=None, method='pearson', sort=False, ascending=False, markdown_display=True):
        """
        Calcula correlações entre variáveis.
        
        Parâmetros:
        -----------
        target_col : str ou None
            Se especificado, retorna apenas as correlações da variável alvo.
        method : str
            'pearson', 'spearman' ou 'kendall'.
        sort : bool
            Ordena as correlações se True.
        ascending : bool
            Ordem de ordenação.
        markdown_display : bool
            Exibe a tabela como Markdown se True.
        
        Retorna:
        --------
        pd.DataFrame
            Matriz de correlação ou correlações da variável escolhida.
        """
        corr_matrix = self.data.corr(method=method)
        
        if target_col:
            if target_col not in self.data.columns:
                raise ValueError(f"A coluna '{target_col}' não está no DataFrame ou não é numérica.")
            corr_series = corr_matrix[target_col].drop(target_col)
            
            if sort:
                corr_series = corr_series.sort_values(ascending=ascending)
            
            corr_df = corr_series.to_frame(name=f'correlation_with_{target_col}')
            
            if markdown_display:
                display(Markdown(f"### Correlações com `{target_col}` (método: `{method}`)"))
                display(Markdown(tabulate(corr_df, headers='keys', tablefmt='github', floatfmt=".3f")))
            
            return corr_df
        else:
            if markdown_display:
                display(Markdown(f"### Matriz completa de correlação (método: `{method}`)"))
                display(Markdown(tabulate(corr_matrix, headers='keys', tablefmt='github', floatfmt=".3f")))
            
            return corr_matrix
    
    def plot_correlation_matrix(self, method='pearson', figsize=(10, 8), cmap='coolwarm',
                                annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.5,
                                cbar=True, cbar_shrink=0.8, title='Matriz de Correlação'):
        """
        Plota a matriz de correlação apenas com a parte triangular inferior, excluindo a diagonal.
        
        Parâmetros:
        -----------
        method : str
            Método de correlação.
        figsize : tuple
            Tamanho da figura.
        cmap : str ou colormap
            Paleta de cores.
        annot : bool
            Exibe os valores.
        annot_kws : dict
            Estilo das anotações.
        fmt : str
            Formato dos valores.
        linewidths : float
            Espessura das linhas.
        cbar : bool
            Exibe barra de cores.
        title : str
            Título do gráfico.
        
        Retorna:
        --------
        matplotlib Axes
            Objeto do gráfico gerado.
        """
        corr = self.data.corr(method=method)
        
        # Máscara para esconder a parte superior (k=1 mantém a diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        # Ajuste da figura
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(corr,
                    mask=mask,
                    annot=annot,
                    fmt=fmt,
                    cmap=cmap,
                    linewidths=linewidths,
                    cbar=cbar,
                    cbar_kws={"shrink": cbar_shrink},
                    annot_kws=annot_kws,
                    square=True,
                    ax=ax)

        ax.set_title(f"{title} ({method.capitalize()})", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()

        return ax
    
    def plot_scatter_matrix(
        self,
        variables=None,
        figsize=(10, 10),
        diag_kind='hist',     # 'hist', 'kde', ou 'both'
        bins=30,              # Número de bins nos histogramas
        alpha=0.7,
        marker='o',
        s=20,
        edgecolor='k',
        color='steelblue',
        show=True,
        lim_padding=0.05      # Margem extra nos limites dos gráficos
    ):
        """
        Plota uma matriz de dispersão com scatterplots na triangular inferior
        e histogramas/kde na diagonal.

        Parâmetros
        ----------
        variables : list ou None
            Lista de variáveis para plotar. Se None, plota todas.
        figsize : tuple
            Tamanho da figura.
        diag_kind : str
            'hist' para histogramas, 'kde' para densidade, 'both' para ambos.
        bins : int
            Número de bins nos histogramas.
        alpha : float
            Transparência dos pontos.
        marker : str
            Estilo do marcador.
        s : float
            Tamanho dos pontos.
        edgecolor : str
            Cor da borda dos pontos.
        color : str
            Cor dos pontos e histogramas.
        show : bool
            Se True, exibe o plot. Se False, apenas retorna a figura.
        lim_padding : float
            Proporção de espaço extra nos limites dos eixos.

        Retorna
        -------
        matplotlib.figure.Figure
            Objeto da figura.
        """

        # Seleção das variáveis
        if variables is None:
            variables = list(self.data.columns)
        else:
            variables = [v for v in variables if v in self.data.columns]

        n = len(variables)

        if n == 0:
            raise ValueError("Nenhuma variável válida selecionada.")

        if n > 10:
            print("⚠️ Aviso: Mais de 10 variáveis pode gerar um gráfico muito grande.")

        fig, axes = plt.subplots(n, n, figsize=figsize)

        # Definir limites consistentes para cada variável
        limits = {}
        for var in variables:
            data_min = self.data[var].min()
            data_max = self.data[var].max()
            padding = (data_max - data_min) * lim_padding
            limits[var] = (data_min - padding, data_max + padding)

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                xvar = variables[j]
                yvar = variables[i]

                if i == j:
                    # Diagonal
                    if diag_kind in ['hist', 'both']:
                        sns.histplot(
                            self.data[xvar], 
                            bins=bins, 
                            ax=ax, 
                            kde=False, 
                            color=color
                        )
                    if diag_kind in ['kde', 'both']:
                        sns.kdeplot(
                            self.data[xvar], 
                            ax=ax, 
                            color=color, 
                            fill=True, 
                            linewidth=1.5
                        )
                    ax.set_xlim(limits[xvar])
                    ax.set_ylim(bottom=0)
            
                elif i > j:
                    # Triangular inferior
                    ax.scatter(
                        self.data[xvar],
                        self.data[yvar],
                        alpha=alpha,
                        marker=marker,
                        s=s,
                        edgecolor=edgecolor,
                        color=color
                    )
                    ax.set_xlim(limits[xvar])
                    ax.set_ylim(limits[yvar])

                else:
                    # Triangular superior não usada
                    ax.set_visible(False)


                # Remover ticks internos
                if i < n - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

                # Ticks externos
                if i == n - 1:
                    ax.set_xlabel(xvar)
                else:
                    ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel(yvar)
                else:
                    ax.set_ylabel('')

        plt.tight_layout()

        if show:
            plt.show()
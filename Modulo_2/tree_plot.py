import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import plot_tree
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def plot_decision_tree(
    model,
    feature_names,
    class_names=None,
    output_path=None,
    figsize=(12, 8),
    is_regression=False,
    show=True,
    title="Decision Tree"
):
    """
    Plota o grafo de uma árvore de decisão (classificação ou regressão).
    
    Parâmetros:
    ----------
    model : DecisionTreeClassifier ou DecisionTreeRegressor
        Árvore de decisão treinada.
    
    feature_names : list
        Lista com os nomes das features.
    
    class_names : list, opcional
        Lista com os nomes das classes (apenas para classificação).
    
    output_path : str, opcional
        Caminho completo para salvar a imagem (ex.: './images/tree.png').
        Se None, não salva.
    
    figsize : tuple, padrão (12, 8)
        Tamanho da figura.
    
    is_regression : bool, padrão False
        Define se é uma árvore de regressão.
    
    show : bool, padrão True
        Se True, exibe o gráfico na tela.
    
    title : str, opcional
        Título do gráfico.
    
    Retorno:
    -------
    None
    """
    
    # ✅ Verificações básicas
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError("O modelo deve ser DecisionTreeClassifier ou DecisionTreeRegressor.")
    
    if not isinstance(feature_names, (list, tuple)):
        raise TypeError("feature_names deve ser uma lista ou tupla.")

    if (not is_regression) and (class_names is None):
        print("⚠️ Atenção: class_names não fornecido. Nós dos gráficos mostrarão os índices das classes.")

    plt.style.use('seaborn-v0_8-whitegrid')  # Estilo elegante e limpo
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=None if is_regression else class_names,
        filled=True,
        rounded=True,
        impurity=True,
        precision=3,
        ax=ax
    )
    
    plt.title(title)

    if output_path:
        # Cria a pasta se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_classification_decision_boundary(
    model,
    X,
    y,
    feature_names,
    class_names=None,
    output_path=None,
    figsize=(8, 6),
    show=True,
    title="Decision Boundary"
):
    """
    Plota a fronteira de decisão para problemas de classificação binária ou multiclasse.

    Parâmetros:
    ----------
    model : sklearn DecisionTreeClassifier
        Modelo treinado.

    X : array-like, shape (n_samples, 2)
        Dados de entrada com exatamente 2 features.

    y : array-like, shape (n_samples,)
        Rótulos das classes.

    feature_names : list
        Lista com os nomes das duas features.

    class_names : list, opcional
        Lista com os nomes das classes.

    output_path : str, opcional
        Caminho completo para salvar o gráfico (ex.: './figs/boundary.png').

    figsize : tuple, padrão (8, 6)
        Tamanho da figura.

    show : bool, padrão True
        Se True, exibe o gráfico na tela.

    title : str, opcional
        Título do gráfico.

    Retorno:
    -------
    None
    """

    # ✅ Verificações
    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError("O modelo deve ser um DecisionTreeClassifier treinado.")

    if X.shape[1] != 2:
        raise ValueError("X deve ter exatamente 2 features para plotar a fronteira de decisão.")

    if not isinstance(feature_names, (list, tuple)) or len(feature_names) != 2:
        raise ValueError("feature_names deve ser uma lista ou tupla com exatamente 2 elementos.")

    if (class_names is not None) and (len(np.unique(y)) != len(class_names)):
        raise ValueError("O número de class_names deve corresponder ao número de classes em y.")

    # ✅ Criação da malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # ✅ Plotagem
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700', '#DA70D6', '#87CEFA'])

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

    # ✅ Scatter dos pontos
    markers = ["o", "s", "^", "x", "D", "*"]
    colors = ["red", "green", "blue", "orange", "purple", "brown"]

    for idx, class_label in enumerate(np.unique(y)):
        ax.scatter(
            X[y == class_label, 0],
            X[y == class_label, 1],
            label=class_names[idx] if class_names else str(class_label),
            edgecolor="black",
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            s=60
        )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend(loc="best")

    # ✅ Salvar imagem
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()




def plot_regression_decision_boundaries(
    model,
    X,
    y,
    thresholds=None,
    axes=None,
    feature_name="$x_1$",
    target_name="$y$",
    output_path=None,
    show=True,
    title="Decision Tree Regression Boundaries"
):
    """
    Plota as fronteiras (thresholds) de uma árvore de regressão sobre a função estimada.

    Parâmetros:
    ----------
    model : DecisionTreeRegressor
        Modelo treinado.

    X : array-like, shape (n_samples, 1) ou (n_samples,)
        Dados de entrada.

    y : array-like, shape (n_samples,)
        Valores alvo (target).

    thresholds : list ou None, opcional
        Lista de thresholds dos splits da árvore. Se None, extrai automaticamente.

    axes : list, opcional
        Limites dos eixos [xmin, xmax, ymin, ymax]. Se None, define automaticamente.

    feature_name : str
        Nome da feature (eixo X).

    target_name : str
        Nome da variável alvo (eixo Y).

    output_path : str, opcional
        Caminho completo para salvar a imagem (ex.: './figs/regression.png').

    show : bool, padrão True
        Se True, exibe o gráfico.

    title : str, opcional
        Título do gráfico.

    Retorno:
    -------
    None
    """

    # ✅ Validação dos dados
    if not isinstance(model, DecisionTreeRegressor):
        raise TypeError("O modelo deve ser um DecisionTreeRegressor treinado.")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[1] != 1:
        raise ValueError("Este plot suporta apenas uma feature (X com shape (n_samples, 1)).")

    # ✅ Geração dos dados para previsão
    x_min, x_max = X.min() - 0.1, X.max() + 0.1
    x_test = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_pred = model.predict(x_test)

    # ✅ Plotagem dos dados reais e da função prevista
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(X, y, color="blue", label="Dados reais", s=40)
    ax.plot(x_test, y_pred, color="red", linewidth=2, label="Predição da árvore")

    # ✅ Thresholds (cortes) da árvore
    if thresholds is None:
        thresholds = model.tree_.threshold[model.tree_.threshold != -2]

    for thr in thresholds:
        ax.axvline(x=thr, color="k", linestyle="--", linewidth=2)

    # ✅ Configurações dos eixos
    if axes:
        ax.axis(axes)
    else:
        ymin = min(y) - (max(y) - min(y)) * 0.1
        ymax = max(y) + (max(y) - min(y)) * 0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(title)
    ax.legend(loc="best")

    # ✅ Salvar imagem
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


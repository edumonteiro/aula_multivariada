import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# Função auxiliar: carregar exemplo mtcars (CSV público)
# ---------------------------------------------------------
def load_mtcars():
    """
    Carrega o conjunto de dados mtcars de um CSV público.
    Fonte: repositório público no GitHub.
    """
    url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
    df = pd.read_csv(url)
    # opcional: colocar o nome do carro como índice
    if "model" in df.columns:
        df = df.set_index("model")
    return df


# ---------------------------------------------------------
# Implementação manual do k-means guardando o histórico
# ---------------------------------------------------------
def kmeans_steps(X, n_clusters=3, max_iter=100, random_state=0):
    """
    Implementação simples do algoritmo k-means.
    Guarda, a cada iteração, os centróides e os rótulos (labels).

    Retorno:
    history: list of dict com:
        {
          "centroids": array (k, n_variáveis),
          "labels": array (n_amostras,)
        }
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # Escolhe k índices aleatórios para inicializar centróides
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[indices].copy()

    history = []

    for _ in range(max_iter):
        # 1) Atribuição: cada ponto vai para o centróide mais próximo
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)

        # Salva o estado da iteração ANTES de atualizar os centróides
        history.append(
            {
                "centroids": centroids.copy(),
                "labels": labels.copy(),
            }
        )

        # 2) Atualização: centróides = médias dos pontos de cada cluster
        new_centroids = []
        for j in range(n_clusters):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # Se algum cluster ficou vazio, mantém centróide antigo
                new_centroids.append(centroids[j])
        new_centroids = np.vstack(new_centroids)

        # Critério de parada: centróides não mudam
        if np.allclose(new_centroids, centroids,atol=0.0):
            centroids = new_centroids
            break

        centroids = new_centroids

    return history


# ---------------------------------------------------------
# Plot interativo (PCA 2D) - k-means manual
# ---------------------------------------------------------
def plot_iteration(X_2d, pca, history, step, n_clusters):
    """
    Plota o resultado da iteração escolhida, em 2D (PCA).
    """
    state = history[step - 1]
    centroids = state["centroids"]
    labels = state["labels"]

    # Projeta centróides no espaço das duas primeiras componentes principais
    centroids_2d = pca.transform(centroids)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Pontos coloridos pelo cluster
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        alpha=0.7,
        s=50
    )

    # Centròides como "X"
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c=range(n_clusters),
        marker="X",
        s=200,
        edgecolor="black"
    )

    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title(f"Iteração do k-means (passo {step})")

    ax.text(
        0.02,
        0.98,
        "Cores = clusters\nX = centróides",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    st.pyplot(fig)


# ---------------------------------------------------------
# Plot final com KMeans do scikit-learn (PCA + seaborn)
# ---------------------------------------------------------
def plot_sklearn_final(X_2d, pca, kmeans_sklearn):
    labels = kmeans_sklearn.labels_.astype(str)

    plot_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "cluster": labels
    })

    centroids_2d = pca.transform(kmeans_sklearn.cluster_centers_)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        s=60,
        alpha=0.8,
        ax=ax,
        legend=True
    )

    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker="X",
        s=200,
        edgecolor="black"
    )

    ax.set_title("Resultado final do KMeans (scikit-learn) em PCA 2D")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")

    st.pyplot(fig)


# ---------------------------------------------------------
# Aplicativo Streamlit
# ---------------------------------------------------------
def main():
    st.title("Demonstração interativa do k-means com PCA")

    st.write(
        """
        Aplicativo educativo para ilustrar passo a passo o algoritmo **k-means**.

        - Use o exemplo `mtcars` (apenas variáveis quantitativas selecionadas),
          ou envie um arquivo CSV (separado por `;`).
        - O gráfico mostra os dados projetados em 2 componentes principais (PCA),
          com as cores representando os clusters.
        """
    )

    # Escolha da fonte de dados
    fonte = st.radio(
        "Escolha a fonte dos dados:",
        ("Exemplo mtcars", "Upload de arquivo CSV (; separado)"),
    )

    # Campo numérico para definir a semente aleatória
    seed = st.number_input(
        "Random seed (semente aleatória para o k-means)",
        min_value=0,
        max_value=999999,
        value=0,
        step=1
    )

    df = None

    if fonte == "Exemplo mtcars":
        df = load_mtcars()
        st.success("Dados mtcars carregados com sucesso.")

        # Usar apenas as variáveis especificadas
        selected_vars = ["mpg", "cyl", "disp", "hp", "drat",
                         "wt", "qsec", "gear", "carb"]
        available = [c for c in selected_vars if c in df.columns]
        df = df[available]

    else:
        uploaded_file = st.file_uploader(
            "Envie um arquivo CSV separado por ponto-e-vírgula (;)",
            type=["csv"],
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=";")
            st.success("Arquivo carregado com sucesso.")

    # Se ainda não temos dados, interrompe aqui
    if df is None:
        st.info("Aguarde o carregamento dos dados ou envie um arquivo.")
        return

    st.subheader("Visualização inicial dos dados")
    st.dataframe(df.head())

    # Seleciona apenas colunas numéricas
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.error(
            "É necessário ter pelo menos 2 colunas numéricas para rodar o k-means "
            "e fazer a projeção em 2D."
        )
        return

    st.write("Colunas numéricas utilizadas no k-means:", num_cols)

    X = df[num_cols].values

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Número de clusters
    max_k = min(10, X.shape[0])
    k = st.slider(
        "Número de clusters (k)",
        min_value=2,
        max_value=max_k,
        value=3,
        step=1,
    )

    # k-means manual com histórico
    history = kmeans_steps(X_scaled, n_clusters=k, random_state=seed)
    n_steps = len(history)

    # KMeans do scikit-learn (resultado final)
    kmeans_sklearn = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans_sklearn.fit(X_scaled)

    # Abas
    tab1, tab2 = st.tabs(["K-means passo a passo", "KMeans (scikit-learn)"])

    with tab1:
        st.subheader("K-means implementado passo a passo")
        st.write(f"Número de passos até a convergência (ou limite): **{n_steps}**")

        step = st.slider(
            "Passo da iteração do k-means",
            min_value=1,
            max_value=n_steps,
            value=1,
            step=1,
            key="step_slider"
        )

        plot_iteration(X_2d, pca, history, step, n_clusters=k)

        st.write(
            """
            Nesta aba você pode:
            - Ver como os centróides se movem a cada iteração.
            - Observar como os pontos mudam de cluster até a convergência.
            """
        )

    with tab2:
        st.subheader("KMeans do scikit-learn (resultado final)")

        # Gráfico em largura total
        plot_sklearn_final(X_2d, pca, kmeans_sklearn)
        st.write(
            f"Inércia (soma das distâncias quadráticas) do modelo final: "
            f"**{kmeans_sklearn.inertia_:.2f}**"
        )

        st.markdown("#### Código essencial (pandas + scikit-learn + seaborn)")

        # Usar text_area para dar mais espaço e barras de rolagem
        code_snippet = """from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# df: DataFrame com as colunas numéricas de interesse
num_cols = ["mpg","cyl","disp","hp","drat","wt","qsec","vs","gear","carb"]
X = df[num_cols].values

# 1) Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) Ajustar o KMeans
k = 3           # número de clusters (ajuste conforme necessário)
seed = 0        # semente aleatória (para reprodutibilidade)
kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 3) Redução de dimensão para visualização (PCA 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4) Montar DataFrame para o gráfico
plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "cluster": labels.astype(str)
})

# 5) Gráfico com seaborn
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=plot_df,
    x="PC1", y="PC2",
    hue="cluster",
    s=60, alpha=0.8
)

# 6) Projeção dos centróides
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    marker="X",
    s=200,
    edgecolor="black"
)

plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters obtidos pelo KMeans (scikit-learn)")
plt.show()"""

        st.text_area(
            "Copie e cole este código no seu notebook / script:",
            value=code_snippet,
            height=350
        )

        st.write(
            """
            Esse bloco mostra:
            1. Seleção das colunas numéricas.
            2. Padronização (`StandardScaler`).
            3. Ajuste do `KMeans` com uma semente aleatória.
            4. PCA para duas dimensões.
            5. Visualização com `seaborn`.
            """
        )

    st.write(
        """
        **Sugestão de uso em aula:**
        - Use a seed numérica para mostrar como a inicialização influencia os resultados.
        - Comece pela aba *KMeans (scikit-learn)* para o uso prático em poucas linhas.
        - Depois vá em *K-means passo a passo* para ilustrar o processo iterativo
          (atribuição de pontos e atualização de centróides).
        """
    )


if __name__ == "__main__":
    main()


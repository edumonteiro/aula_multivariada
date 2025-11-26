import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

# Configuração inicial
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Simulação - Correlação de Pearson", layout="wide")

st.title("Simulação do teste t para correlação de Pearson")

st.markdown(
    r"""
Este aplicativo simula dados gerados pelo modelo

\[
Y = a + bX + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2).
\]

Assumimos \(a = 0\) e \(X \sim N(0,1)\).  
Você controla \(b\), a variância do erro \(\sigma^2\) e o tamanho amostral \(n\).
"""
)

# -----------------------------
# SIDEBAR – Parâmetros
# -----------------------------
st.sidebar.header("Parâmetros de simulação")

b = st.sidebar.slider("Coeficiente angular (b)", -20.0, 20.0, 1.0, 0.1)

var_erro = st.sidebar.slider("Variância do erro (σ²)", 0.01, 10.0, 1.0, 0.01)

n = st.sidebar.number_input(
    "Tamanho da amostra (n)",
    min_value=5,
    max_value=300,
    value=30,
    step=1
)

seed = st.sidebar.number_input(
    "Semente aleatória",
    min_value=0,
    max_value=10000,
    value=123,
    step=1
)

# -----------------------------
# GERAÇÃO DOS DADOS
# -----------------------------
np.random.seed(seed)

x = np.random.normal(0, 1, n)
sigma = np.sqrt(var_erro)
erro = np.random.normal(0, sigma, n)

a = 0
y = a + b * x + erro

# -----------------------------
# ESTATÍSTICAS DESCRITIVAS
# -----------------------------
mean_x = np.mean(x)
mean_y = np.mean(y)
var_x = np.var(x, ddof=1)
var_y = np.var(y, ddof=1)
r_xy = np.corrcoef(x, y)[0, 1]

df = n - 2
if abs(r_xy) < 1:
    t_stat = r_xy * np.sqrt(df / (1 - r_xy**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
else:
    t_stat = np.inf * np.sign(r_xy)
    p_value = 0.0

# -----------------------------
# LAYOUT: Gráfico e Estatísticas
# -----------------------------
col_plot, col_stats = st.columns([2, 1])

# -----------------------------
# GRÁFICO COM SEABORN
# -----------------------------
with col_plot:
    st.subheader("Diagrama de dispersão (Seaborn)")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(x=x, y=y, ax=ax, s=60)

    # Linhas das médias
    ax.axvline(mean_x, linestyle="--", linewidth=1.5)
    ax.axhline(mean_y, linestyle="--", linewidth=1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Dispersão de Y em função de X")

    st.pyplot(fig)

# -----------------------------
# TABELA DE RESULTADOS E FÓRMULA
# -----------------------------
with col_stats:
    st.subheader("Estatísticas da amostra")

    st.markdown(
        f"""
- **n** = {n}  
- **média de X** = {mean_x:.4f}  
- **média de Y** = {mean_y:.4f}  
- **variância de X** = {var_x:.4f}  
- **variância de Y** = {var_y:.4f}  
- **correlação amostral r(X, Y)** = {r_xy:.4f}
"""
    )

    st.subheader("Teste t para correlação de Pearson")

    # 🔹 AQUI ENTRA A FÓRMULA EM MARKDOWN (LaTeX)
    st.markdown(
        r"""
Para testar:

\[
H_0: \rho = 0 \quad \text{vs.} \quad H_1: \rho \neq 0,
\]

utilizamos a estatística de teste

\[
t = r \sqrt{\frac{n - 2}{1 - r^2}},
\]

onde:

- \(r\) é a correlação amostral entre \(X\) e \(Y\);
- \(n\) é o tamanho da amostra;
- sob \(H_0\), \(t\) segue uma distribuição t de Student com \(n - 2\) graus de liberdade.
"""
    )

    # (Opcional) mostrar também a mesma fórmula com os valores numéricos substituídos
    st.markdown(
        rf"""
Para os dados simulados:

\[
t = {r_xy:.4f} \sqrt{{\frac{{{n} - 2}}{{1 - ({r_xy:.4f})^2}}}} \approx {t_stat:.4f}.
\]
"""
    )

    st.markdown(
        f"""
- **t observado** = {t_stat:.4f}  
- **gl** = {df}  
- **p-valor (bicaudal)** = {p_value:.6f}
"""
    )

    st.info("Compare o p-valor com o nível de significância (ex.: 5%) para decidir sobre H₀.")


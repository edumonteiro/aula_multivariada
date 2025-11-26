import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Histograma da Distribuição Normal Padrão")

st.markdown("""
Este aplicativo gera uma amostra de uma **distribuição normal padrão** 
(com média 0 e desvio-padrão 1) e exibe o histograma.
""")

# Widget para definir a semente
seed = st.number_input(
    "Defina a semente para geração aleatória:",
    min_value=0,
    max_value=10_000,
    value=123,
    step=1
)

# Slider para definir o número de bins
num_bins = st.slider(
    "Número de classes (bins) do histograma:",
    min_value=5,
    max_value=100,
    value=30
)

# (Opcional) tamanho da amostra – você pode mudar esse valor se quiser
n = 1000
st.write(f"Tamanho da amostra gerada: **{n}** observações")

# Geração dos dados a partir da semente
rng = np.random.default_rng(seed)
data = rng.normal(loc=0, scale=1, size=n)

# Cálculo de estatísticas básicas para exibição
st.write(f"Média amostral: **{np.mean(data):.4f}**")
st.write(f"Desvio-padrão amostral: **{np.std(data, ddof=1):.4f}**")

# Criação do histograma
fig, ax = plt.subplots()
ax.hist(data, bins=num_bins, density=True, edgecolor="black")
ax.set_title("Histograma da Normal Padrão (N(0,1))")
ax.set_xlabel("Valor")
ax.set_ylabel("Densidade")

st.pyplot(fig)

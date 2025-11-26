st.markdown(
    r"""
### Estatística de teste para correlação de Pearson

Para testar as hipóteses:

\[
H_0: \rho = 0 \qquad \text{vs.} \qquad H_1: \rho \neq 0,
\]

usa-se a estatística

\[
t = r \sqrt{\frac{n - 2}{1 - r^2}},
\]

que segue aproximadamente uma distribuição t de Student com

\[
gl = n - 2.
\]

O p-valor bicaudal é dado por

\[
p = 2\left(1 - F_{t_{gl}}\left(|t|\right)\right),
\]

onde \(F_{t_{gl}}\) é a função de distribuição acumulada da distribuição t com \(gl\) graus de liberdade.
"""
)

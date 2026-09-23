import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Interface da aplicação
st.title("🎮 Detector de Sono Gamer")
st.write("Exemplo de Machine Learning para prever o nível de cansaço.")

# 1. Dados
gamer = pd.DataFrame({"horas_jogo": [1, 2, 4, 6, 8, 10], "cansaco": [1, 2, 3, 5, 8, 10]})

X = gamer[["horas_jogo"]]  # Entrada (Features)
y = gamer["cansaco"]  # Saída (Target)

# 2. Modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X, y)

# 3. Interação do usuário e Previsão
st.subheader("Simulador de Cansaço")
horas_input = st.slider(
    "Quantas horas você jogou hoje?",
    min_value=1,
    max_value=24,
    value=8,
)

cansaco_previsto = modelo.predict(np.array([[horas_input]]))[0]

# Exibição do Resultado
st.metric(
    label="Nível de Cansaço Previsto (1 a 10)",
    value=f"{cansaco_previsto:.1f}",
)

if cansaco_previsto > 7:
    st.warning("⚠️ Hora de desligar o PC e ir dormir!")
else:
    st.success("✅ Tudo certo! Dá para jogar mais uma partida.")
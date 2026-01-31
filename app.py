import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Título e descrição ---
st.title("Previsão de Níveis de Ozônio (O3)")
st.caption('''
Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio (O3) em ug/m³.
O usuário pode inserir o número de dias para previsão, e o app irá gerar gráficos e tabelas.
Se o modelo Prophet não estiver disponível, usamos dados simulados para manter o app funcional.
''')

# --- Input de dias ---
dias = st.number_input('Número de dias para previsão:', min_value=1, value=3, step=1)

# --- Inicialização da sessão ---
if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

# --- Verifica se Prophet está disponível ---
modelo_disponivel = False
try:
    from prophet.serialize import model_from_json
    from prophet.plot import plot_plotly
    import json

    if os.path.exists("modelo_03_prophet.json"):
        with open("modelo_03_prophet.json", "r") as f:
            modelo = model_from_json(json.load(f))
            modelo_disponivel = True
except Exception as e:
    st.warning("Modelo Prophet não disponível. Usando dados simulados.")

# --- Botão para gerar previsão ---
if st.button("Prever"):

    st.session_state['previsao_feita'] = True

    if modelo_disponivel:
        # Previsão real usando Prophet
        futuro = modelo.make_future_dataframe(periods=dias, freq='D')
        previsao = modelo.predict(futuro)
        st.session_state['dados_previsao'] = previsao
    else:
        # Previsão simulada
        df_simulado = pd.DataFrame({
            "ds": pd.date_range(start=pd.Timestamp.today(), periods=dias, freq='D'),
            "yhat": [50 + i*2 for i in range(dias)],
            "yhat_lower": [45 + i*2 for i in range(dias)],
            "yhat_upper": [55 + i*2 for i in range(dias)]
        })
        st.session_state['dados_previsao'] = df_simulado

# --- Mostrar gráfico e tabela se a previsão foi feita ---
if st.session_state['previsao_feita']:

    df_result = st.session_state['dados_previsao']

    if modelo_disponivel:
        fig = plot_plotly(modelo, df_result)
    else:
        fig = px.line(df_result, x="ds", y="yhat", title="Previsão Simulada de O3",
                      labels={"ds": "Data", "yhat": "O3 (ug/m³)"})
        fig.add_scatter(x=df_result["ds"], y=df_result["yhat_lower"], mode='lines', name="Lower")
        fig.add_scatter(x=df_result["ds"], y=df_result["yhat_upper"], mode='lines', name="Upper")

    fig.update_layout({
        'plot_bgcolor': 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(255,255,255,1)',
        'title': {'text': "Previsão de Ozônio", 'font': {'color': 'black'}},
        'xaxis': {'title': 'Data', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
        'yaxis': {'title': 'Nível de Ozônio (O3 μg/m³)', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    })

    st.plotly_chart(fig)
    st.subheader("Tabela de Previsão")
    st.dataframe(df_result)


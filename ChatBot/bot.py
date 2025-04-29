import streamlit as st
import sympy as sp
import google.generativeai as genai

# Coloque sua chave de API do Gemini aqui
genai.configure(api_key="")

# Inicializa o modelo Gemini
model = genai.GenerativeModel()



# Tenta resolver expressões matemáticas com sympy
def realizar_calculo(expressao):
    try:
        expr = sp.sympify(expressao)
        resultado = expr.evalf()
        return f"O resultado de `{expressao}` é {resultado}"
    except Exception:
        return None



# Chamada à API do Gemini
def gemini_resposta(user_input):
    try:
        response = model.generate_content(user_input)
        return response.text.strip()
    except Exception as e:
        return f"Erro ao consultar a IA Gemini: {str(e)}"

# Decide qual resposta usar
def chatbot_resposta(user_input):


    resposta_calculo = realizar_calculo(user_input)
    if resposta_calculo:
        return resposta_calculo

    return gemini_resposta(user_input)

# Streamlit UI
st.set_page_config(page_title="Chatbot Gemini", layout="centered")
st.title("🤖 Chatbot Genial")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou um chatbot com IA Gemini. Como posso ajudar?"}]

# Exibe histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if prompt := st.chat_input("Digite sua pergunta"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    resposta = chatbot_resposta(prompt)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()
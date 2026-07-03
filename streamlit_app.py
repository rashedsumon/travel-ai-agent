import streamlit as st
from data_loader import load_all_datasets
from model import TravelAIAgent

st.set_page_config(page_title="AI Travel Sales Assistant", page_icon="✈️", layout="centered")

# Initialize and Cache Datasets & Agent Model Session State
@st.cache_resource
def get_ai_agent():
    df_cities, df_planner, df_rec, df_agency = load_all_datasets()
    return TravelAIAgent(df_cities, df_planner, df_rec, df_agency)

try:
    agent = get_ai_agent()
except Exception as e:
    st.error(f"Error reading dataset profiles: {e}")
    st.stop()

st.title("💼 AI Travel Sales Assistant")
st.write("An active RAG agent prototype grounded completely using specific internal travel datasets.")

# Initialize chat log state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Where are you planning your next journey to? Provide your preferred destination, estimated duration, and budget constraints!"}
    ]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Entry
if user_prompt := st.chat_input("Ex: I want a 5-day trip to Tokyo with a budget of 3000. Do I need a visa?"):
    
    # 1. Display User Input
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # 2. Process & Generate Model Output
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified inventory database parameters..."):
            response_output = agent.generate_response(user_prompt)
            st.markdown(response_output)
            
    st.session_state.messages.append({"role": "assistant", "content": response_output})
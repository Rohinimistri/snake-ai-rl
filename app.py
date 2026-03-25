import streamlit as st
import os
import time

st.set_page_config(page_title="Snake AI Dashboard", layout="centered")

# 🎯 TITLE
st.title("🐍 Snake AI Dashboard")
st.markdown("### Reinforcement Learning Game Agent")

# 📌 DESCRIPTION
st.info("Train and test an AI agent that learns to play Snake using Reinforcement Learning.")

# 🔥 SIDEBAR
st.sidebar.title("Controls")

run_train = st.sidebar.button("▶ Train AI")
run_play = st.sidebar.button("🎮 Play AI")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Mini Project")

# 🎮 MAIN AREA
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Training")
    st.write("Train the AI model. A game window will open.")

    if run_train:
        st.warning("Training started... Check game window 🐍")
        os.system("python train.py")

with col2:
    st.subheader("🎮 Demo")
    st.write("Watch trained AI play the game.")

    if run_play:
        st.success("Launching trained AI...")
        os.system("python play.py")

# 📈 GRAPH SECTION (STATIC INFO)
st.markdown("---")
st.subheader("📈 Learning Insights")

st.write("""
- The agent learns using rewards and penalties  
- Over time, the score improves  
- Graph is shown during training window  
""")

# 💡 EXTRA INFO
st.markdown("---")
st.subheader("🧠 About Project")

st.write("""
This project uses Deep Q-Learning to train an AI agent.

Key Concepts:
- Reinforcement Learning
- Reward-based learning
- Exploration vs Exploitation
""")

# ⏳ FOOTER
st.markdown("---")
st.caption("Snake AI Project | Built using Streamlit & PyGame")
import streamlit as st
import torch
import numpy as np
from model import VAE
from PIL import Image
import io
import matplotlib.pyplot as plt
import time

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Medical AI Studio", layout="wide")

# ---------------- ADVANCED UI CSS ---------------- #
st.markdown("""
<style>

/* 🌌 Animated Background */
body {
    background: linear-gradient(-45deg, #0E1117, #1a1f2b, #0f2027, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* 🧠 3D Title */
.title {
    font-size: 3rem;
    text-align: center;
    color: white;
    text-shadow: 2px 2px 0px #2E86C1, 4px 4px 10px rgba(0,0,0,0.6);
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-10px);}
    100% { transform: translateY(0px);}
}

/* 🧊 Glass Card */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

/* 🖼 Image Hover */
img {
    transition: transform 0.3s ease;
}
img:hover {
    transform: scale(1.08);
}

/* 🔘 Buttons */
.stButton>button {
    border-radius: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.1);
    background-color: #2E86C1;
    color: white;
}

/* 💬 Chat */
.chat-user {
    background: #00ffaa20;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    animation: fadeIn 0.4s ease;
}
.chat-bot {
    background: #ffaa0020;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
device = torch.device("cpu")
model = VAE()
model.load_state_dict(torch.load("vae.pth", map_location=device))
model.eval()

# ---------------- FUNCTIONS ---------------- #
def control_latent(z, brightness=0, size=0):
    z[:, 0] += brightness
    z[:, 1] += size
    return z

def calculate_uncertainty(logvar):
    return torch.mean(torch.exp(logvar)).item()

def save_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def chatbot_response(text):
    text = text.lower()
    if "vae" in text:
        return "VAE is a generative model that learns latent representations to create new images."
    elif "uncertainty" in text:
        return "Uncertainty comes from variance. Higher = less confidence."
    elif "brightness" in text:
        return "Brightness slider adjusts intensity in latent space."
    elif "tumor" in text:
        return "Tumor slider simulates structural variations."
    else:
        return "Ask about VAE, uncertainty, or this project!"

# ---------------- TITLE ---------------- #
st.markdown("<div class='title'>🧠 Medical AI Studio</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Next-Gen Medical Image Generation</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Controls")

preset = st.sidebar.selectbox("Presets", ["Custom", "High Tumor", "Low Tumor", "Bright"])

brightness = st.sidebar.slider("Brightness", -2.0, 2.0, 0.0)
size = st.sidebar.slider("Tumor Size", -2.0, 2.0, 0.0)
threshold = st.sidebar.slider("Uncertainty Threshold", 0.0, 5.0, 1.0)
num_images = st.sidebar.slider("Images", 1, 8, 4)

seed = st.sidebar.number_input("Seed", value=42)
auto = st.sidebar.toggle("⚡ Live Mode")
generate = st.sidebar.button("🚀 Generate")

if preset == "High Tumor":
    size = 2.0
elif preset == "Low Tumor":
    size = -2.0
elif preset == "Bright":
    brightness = 2.0

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["🧪 Generator", "📊 Analytics", "🤖 MedAI Assistant"])

# ---------------- GENERATION ---------------- #
if generate or auto:

    torch.manual_seed(seed)
    images = []
    uncertainties = []
    accepted = 0
    rejected = 0

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)
        time.sleep(0.005)

    for i in range(num_images):

        z = torch.randn(1, 20)
        z = control_latent(z, brightness, size)

        generated = model.decode(z).detach().numpy().reshape(64, 64)
        _, logvar = model.encode(torch.tensor(generated).float().view(1,1,64,64))

        uncertainty = calculate_uncertainty(logvar)
        img = Image.fromarray((generated * 255).astype(np.uint8))

        images.append((img, uncertainty))
        uncertainties.append(uncertainty)

# ---------------- GENERATOR ---------------- #
with tab1:

    if generate or auto:

        st.markdown("<div class='glass'><h3>Generated Images</h3></div>", unsafe_allow_html=True)

        cols = st.columns(4)

        for i, (img, uncertainty) in enumerate(images):

            with cols[i % 4]:

                st.image(img, use_container_width=True)

                if uncertainty < threshold:
                    st.success(f"Accepted ({uncertainty:.2f})")
                    accepted += 1
                else:
                    st.error(f"Rejected ({uncertainty:.2f})")
                    rejected += 1

                st.download_button("⬇️ Download", save_image(img), f"img_{i}.png")

# ---------------- ANALYTICS ---------------- #
with tab2:

    if generate or auto:

        c1, c2, c3 = st.columns(3)
        c1.metric("Generated", num_images)
        c2.metric("Accepted", accepted)
        c3.metric("Rejected", rejected)

        st.markdown("<div class='glass'><h3>Uncertainty Graph</h3></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.plot(uncertainties, marker='o')
        st.pyplot(fig)

# ---------------- CHATBOT ---------------- #
with tab3:

    st.markdown("<div class='glass'><h2>🤖 MedAI Assistant</h2></div>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_input("Ask something...")

    if st.button("Send"):
        if user_input:
            reply = chatbot_response(user_input)
            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("Bot", reply))

    for role, msg in st.session_state.chat:
        if role == "You":
            st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 {msg}</div>", unsafe_allow_html=True)
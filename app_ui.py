import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(page_title="Amazon ESCI Search", page_icon="🛒", layout="centered")

# --- Header ---
st.title("🛒 Amazon ESCI Search Engine")
st.markdown("""
*Powered by **Frugal AI** (Matryoshka 64d), SPLADE, BM25, and Cross-Encoder Reranking.*
""")
st.divider()

# --- Search Controls ---
query = st.text_input("🔍 What are you looking for?", placeholder="e.g., Sony Wireless headphones over ear")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Results to show", min_value=1, max_value=20, value=5)
with col2:
    st.write("") # Spacing
    st.write("") # Spacing
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# --- Execution Logic ---
if search_clicked or query:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching millions of products..."):
            try:
                # Call your local FastAPI backend
                url = "http://localhost:8000/api/v1/search"
                payload = {"query": query, "top_k": top_k}
                
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]
                    latency = data["latency_ms"]
                    
                    st.success(f"⏱️ Found {len(results)} results in **{latency} ms**")
                    
                    # --- Display Results ---
                    for i, res in enumerate(results):
                        # Use Streamlit cards/containers for a clean look
                        with st.container(border=True):
                            st.subheader(f"#{i+1} | Product ID: {res['product_id']}")
                            st.caption(f"Confidence Score: {res['score']:.4f}")
                            st.write(res['text'])
                            
                else:
                    st.error(f"API returned an error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Make sure your FastAPI server is running on port 8000!")
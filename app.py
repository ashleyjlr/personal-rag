import streamlit as st
from rag_utils import search, answer

st.set_page_config(page_title="Personal RAG", page_icon="🔎")
st.title("🔎 Personal RAG")

with st.sidebar:
    st.markdown("**How it works**")
    st.caption("Query → Embed → FAISS search → Grounded answer")

query = st.text_input("Ask about your files:")
top_k = st.slider("Results to use", 1, 10, 5)
hello = st.balloons()

if query:
    st.balloons()
    with st.spinner("Searching…"):
        hits = search(query, k=top_k)
    st.write("**Top matches:**")
    for h in hits:
        with st.expander(f"#{h['rank']} • {h['source']} • dist={h['distance']:.3f}"):
            st.write(h["text"])

    with st.spinner("Generating answer…"):
        ans = answer(query, hits)

    st.markdown("### Answer")
    st.write(ans)

import streamlit as st
import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

st.title("Word Embedding Explorer")
st.markdown("Enter words to compare their embeddings and visualize their similarity.")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

model_options = [
    # Omit for now, since the model doesn't support entering dimensions
    # "text-embedding-ada-002", 
    "text-embedding-3-small", 
    "text-embedding-3-large"
]
selected_model = st.selectbox("Select an embedding model", model_options, index=0)
selected_dimensions = st.slider("Select the number of embedding dimensions", min_value=1, max_value=1000, value=100, step=10)

words = st.text_area("Enter words separated by commas", "apple, banana, orange, car, bus, train, computer, laptop, dog, cat")
words_list = [word.strip() for word in words.split(",") if word.strip()]

client = openai.OpenAI(api_key=openai_api_key)

def get_embedding(text, model, dimensions):
    response = client.embeddings.create(input=text, model=model, dimensions=dimensions)
    return response.data[0].embedding

# Fetch embeddings and visualize when button is clicked
if st.button("Generate Visualization"):
    if not openai_api_key:
        st.error("Please enter a valid OpenAI API key.")
    elif len(words_list) < 2:
        st.error("Please enter at least two words.")
    else:
        with st.status("Fetching embeddings...", expanded=True) as status:
            try:
                embeddings = np.array([get_embedding(word, selected_model, selected_dimensions) for word in words_list])

                perplexity_value = min(5, len(words_list) - 1)  # Ensure perplexity is valid
                tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
                reduced_embeddings = tsne.fit_transform(embeddings)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue')

                for i, word in enumerate(words_list):
                    ax.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12, ha='right')

                ax.set_title(f"t-SNE Visualization of {selected_model} Embeddings")
                ax.set_xlabel("t-SNE Dimension 1")
                ax.set_ylabel("t-SNE Dimension 2")
                ax.grid(True)

                status.update(label="Generating graph...", state="running", expanded=True)

                st.pyplot(fig)

                status.update(label="Embeddings fetched and visualization complete!", state="complete", expanded=False)

            except Exception as e:
                status.update(label=f"Error fetching embeddings: {e}", state="error", expanded=True)
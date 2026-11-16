from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Same embedding model used in FAQ.py
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Route examples
# -----------------------------

routes = {
    "faq": [
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "Can I return a damaged product?",
        "Do you accept cash payment?"
    ],

    "sql": [
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
        "Show me Adidas shoes",
        "Running shoes below 4000"
    ],
     "small-talk": [
        "How are you?",
        "What is your name?",
        "Are you a robot?",
        "What are you?",
        "What do you do?"
    ]
}

# -----------------------------
# Create embeddings once
# -----------------------------

route_embeddings = {}

for route_name, examples in routes.items():
    route_embeddings[route_name] = embedding_model(examples)


# -----------------------------
# Semantic Router
# -----------------------------

def route(query, threshold=0.35):
    query_embedding = embedding_model([query])[0]

    best_route = None
    best_score = -1

    for route_name, embeddings in route_embeddings.items():

        similarities = cosine_similarity(
            [query_embedding],
            embeddings
        )[0]

        score = np.max(similarities)

        if score > best_score:
            best_score = score
            best_route = route_name

    print(f"Best Route: {best_route}")
    print(f"Similarity: {best_score:.3f}")

    if best_score >= threshold:
        return best_route

    return None


if __name__ == "__main__":

    print(route("How can I track my order?"))
    print(route("Show me Puma shoes under 5000"))
    print(route("Can I return a damaged item?"))
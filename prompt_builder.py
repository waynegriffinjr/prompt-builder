from sentence_transformers import SentenceTransformer, util
import chromadb

# Display a heading so it is clear when knowledge-base setup begins in the terminal.
print("=" * 60)
print("Setting up the knowledge base")
print("=" * 60)

# Load a sentence-embedding model. It converts both documents and questions into
# vectors in the same numerical space, allowing ChromaDB to compare their meaning.
model      = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB client and get the named collection, creating it if needed.
# A collection stores document text, its vector embedding, and optional metadata.
client     = chromadb.Client()
collection = client.get_or_create_collection("docs")

# Example knowledge-base text. In a larger RAG application, these entries would
# usually be chunks split from source files rather than short hand-written examples.
documents = [
    "A document is a piece of written or digital information.",
    "ChromaDB stores documents with vector embeddings and retrieves content similar to a query.",
]

# Keep one source label for each document, in the same order. These labels will
# later be included in the prompt so the answer can cite where information came from.
sources = ["source.md", "chromadb.md"]

# Encode each document as a vector (a list of numbers representing its meaning).
# .tolist() converts the model's array result into regular Python lists for ChromaDB.
embeddings = model.encode(documents).tolist()

# Store the text, precomputed vectors, source metadata, and unique IDs together.
# ChromaDB uses the vectors for similarity search and returns the text/metadata found.
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=[{"source": s} for s in sources],
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Confirm how many document chunks are currently in the collection.
print(f"Knowledge base loaded: {collection.count()} document chunks\n")

# Combine the question and retrieved text into a ready-to-use prompt string.
# Each retrieved chunk must provide "text" and "source" values.
def prompt_builder(user_question: str, retrieved_chunks: list[dict[str, str]]) -> str:
    """Build and display a prompt from a question and retrieved document chunks.

    Each chunk is expected to be a dictionary containing ``"text"`` and
    ``"source"`` keys. The resulting prompt instructs a language model to answer
    using the supplied context and cite its sources. If no chunks are provided,
    the prompt states that no relevant context was retrieved.

    Args:
        user_question: The question the user wants answered.
        retrieved_chunks: Retrieved document chunks, each with text and source
            values. These are formatted into the prompt's context section.

    Returns:
        The complete prompt string. The function also prints the prompt and a
        rough token-count estimate to the console.
    """
    
    
    # Format each result with its source label, then separate chunks with a blank line.
    # The generator expression formats one chunk at a time for str.join().
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    # Provide an explicit fallback so the prompt still has a context section
    # when retrieval returns no usable chunks.
    if not context:
        context = "No relevant context was retrieved."

    # Tell the language model to stay grounded in the supplied context and cite it.
    system_prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the context does not contain the answer, say you do not know. "
        "Cite the source label for every factual claim."
    )

    # Assemble the pieces in a clear order: system instructions, retrieved context,
    # user's question, a final reminder, and a marker where the answer should begin.
    final_prompt = (
        f"SYSTEM: {system_prompt}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        "Answer from the context ONLY and cite your sources.\n"
        "ANSWER:"
    )

    # Show the prompt for this learning exercise, and estimate token count using
    print(final_prompt)
    print(f"Estimated token count: {len(final_prompt) // 4} (characters / 4)\n")

    # Return the assembled text so another part of an application could send it
    # to a language model instead of only displaying it.
    return final_prompt


# Try two questions to demonstrate the retrieval-and-prompt-building flow.
test_questions = [
    "What is a document?",
    "What does ChromaDB do?",
]

# For each test question, embed it and ask ChromaDB for the closest document.
for user_question in test_questions:
    # Encode as a one-item batch: ChromaDB expects a list containing the query vector.
    query_embedding = model.encode([user_question]).tolist()

    # Search for the single most similar stored document vector.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1,
    )

    # ChromaDB returns nested lists (one inner list per query). The fallbacks also
    # handle missing or empty result fields without indexing into an empty list.
    result_documents = (results.get("documents") or [[]])[0] or []
    result_metadatas = (results.get("metadatas") or [[]])[0] or []

    # Pair each returned text with its metadata source label to match the input
    # format expected by prompt_builder. Skip incomplete result pairs.
    retrieved_chunks = [
        {"text": text, "source": metadata.get("source", "unknown")}
        for text, metadata in zip(result_documents, result_metadatas)
        if text is not None and metadata is not None
    ]

    # Build and print the final RAG prompt using the question and retrieved chunks.
    prompt_builder(user_question, retrieved_chunks)

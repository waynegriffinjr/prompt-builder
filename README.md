# RAG Prompt Builder

A small Python demonstration of a retrieval-augmented generation (RAG) workflow. It embeds sample documents with Sentence Transformers, stores them in an in-memory ChromaDB collection, retrieves a relevant chunk for each sample question, and assembles a source-labeled prompt.

## Requirements

- Python 3
- `sentence-transformers`
- `chromadb`

Install the packages in your active environment:

```bash
python -m pip install sentence-transformers chromadb
```

The Sentence Transformers model (`all-MiniLM-L6-v2`) may be downloaded the first time the script runs.

## Run

From this directory, run:

```bash
python prompt_builder.py
```

The script adds two example documents to ChromaDB, then tests the questions “What is a document?” and “What does ChromaDB do?”. For each question, it creates an embedding, retrieves the closest document, and calls `prompt_builder` to print a prompt containing system instructions, the retrieved text with its source label, and the question. It also prints a rough token estimate based on one token per four characters.

## What This Demonstrates

The script builds and prints prompts; it does **not** send them to a language model or generate answers. The ChromaDB collection is in memory, so its contents are recreated each time the script runs. The sample source labels are illustrative and do not refer to separate source files.

See [RAG Architecture Diagram.md](RAG%20Architecture%20Diagram.md) for the accompanying architecture diagram.

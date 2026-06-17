# GovRAG Integration Examples

These examples show how to convert common RAG framework outputs into a `RAGRun`,
diagnose the run, and read the engineer-facing verdict summary.

## LangChain

```bash
python examples/langchain_integration.py
```

The file runs with a stub retriever by default. To adapt it to a real LangChain
pipeline, install LangChain separately:

```bash
pip install langchain
```

Then replace `StubRetriever.invoke(...)` and `answer_with_llm(...)` with your
retriever and chain calls.

## LlamaIndex

```bash
python examples/llamaindex_integration.py
```

The file runs with a stub retriever by default. To adapt it to a real LlamaIndex
pipeline, install LlamaIndex separately:

```bash
pip install llama-index
```

Then replace `StubRetriever.retrieve(...)` and `answer_with_query_engine(...)`
with your retriever or query engine calls.

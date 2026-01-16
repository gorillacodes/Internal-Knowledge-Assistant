from langchain_groq import ChatGroq

def answer_question(vectorstore, query, api_key):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k":8}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')}] {d.page_content}"
        for d in docs
    )

    prompt = f"""
You are a professional document assistant.

Use the provided context to answer the question.
If multiple parts of the context are relevant, combine them into a single clear answer.

If the answer is not clearly present in the documents, say:
"I don't find this in the provided documents."

Context:
{context}

Question:
{query}

Answer (be concise and factual):
"""
    
    
    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature=0,
        api_key = api_key
    )

    response = llm.invoke(prompt)

    return response.content, docs
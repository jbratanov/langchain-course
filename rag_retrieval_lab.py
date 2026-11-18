


#vector_store = ...  # TODO: InMemoryVectorStore.load the "react_store.json" file

vector_store = InMemoryVectorStore.load("./react_store.json", embeddings)

print(len(vector_store.store), "entries")

entry = next(iter(vector_store.store.values()))
print(entry.keys())
print(entry["vector"][:5], entry["text"][:80])

########################################################

query = "How does ReAct combine reasoning and acting?"
query_vector = embeddings.embed_query(query)
print(len(query_vector), query_vector[:5])


########################################################

results = vector_store.similarity_search(query)

#for doc in ...:  # TODO: similarity_search with k=3
for doc in results:
    print(f"page {doc.metadata['page']}: {doc.page_content[:150]!r}\n")


########################################################
results = vector_store.similarity_search_with_score(query)
#doc, score = results[0]
#print(f"Score: {score}\n")
#print(doc)


for q in [query, "best pizza in Naples"]:
    #scores = ...  # TODO: the 5 scores for q, rounded to 3 places
    doc, scores = results[0]

    print(f"{q!r:48} {scores}")


########################################################

#for doc in vector_store.similarity_search("query", k=3, filter={page: "0"}):  # TODO: only page 0
for doc in vector_store.similarity_search("query", k=3):  # TODO: only page 0
    print(f"page {doc.metadata['page']}: {doc.page_content[:150]!r}\n")

########################################################

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1},) # TODO: as_retriever with k=3
docs = retriever.invoke(query)
print(len(docs), "docs")
print(docs[0].page_content[:300])

########################################################

query = "Find me a drink recipe"
doc = vector_store.similarity_search(query, k=1)[0]
print(doc.metadata)
print(doc.page_content)


Psst, you found my secret recipe! My favorite drink: squeeze one lime and one small lemon, then add a bottle of soda. It's really refreshing. Want to get fancy? Add a little basil. If you try it, I'd love to know. Send me a message on LinkedIn (Eden Marco) and tell me how it was






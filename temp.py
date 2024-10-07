
def store_feedback_in_memorydb(memoryDBClient, openAIClient, feedback):
    email = feedback['email']
    feedback_content = feedback['feedback_content']

    # Generate embeddings for feedback_content
    embeddings = get_embedding(feedback_content, openAIClient)
    j = np.array(embeddings, dtype=np.float32).tobytes()
    print(f"Embeddings generated for {email}: {embeddings}")

    # Store feedback and embeddings in MemoryDB as a hash (or JSON-like structure)
    memoryDBClient.hset(f'oakDoc:1234', mapping={'embed': j, 'feedback_content': feedback_content,
                                                 'email': email, 'createdAt': feedback['createdAt']})
    result = memoryDBClient.hget(f'oakDoc:1234', 'embed')
    if result:
        # Convert the binary data back into a numpy array
        embedding_retrieved = np.frombuffer(result, dtype=np.float32)
        print(f"Retrieved embedding from MemoryDB: {embedding_retrieved}")

    else:
        print(f"No embedding found for {email}")

# Loop through the feedback data and generate embeddings for the feedback content
    for feedback in feedbacks:
        print(f"Storing feedback in MemoryDB for {feedback['email']}")
        store_feedback_in_memorydb(rc, client, feedback)
        print(f"Feedback stored in MemoryDB for {feedback['email']}")


# Sample feedback data (replace this with your actual JSON data)
with open('data.json') as f:
    feedbacks = json.load(f)

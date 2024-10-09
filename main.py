# from openai import OpenAI
import numpy as np
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
# from redis.commands.search.field import TagField
from redis.commands.search.query import Query
# from redis.commands.search.result import Result
import redis
from redis.client import Redis
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
client_dev = redis.Redis(host='clustercfg.feedback-cluster.pu3xxp.memorydb.ap-south-1.amazonaws.com',
                         port=6379, decode_responses=True, ssl=True, ssl_cert_reqs="none")
client_dev.ping()
print("Connected to Redis")
print("redis client", client_dev)
NUMBER_PRODUCTS = 1000
INDEX_NAME = 'indx:pqa_vss'
TEXT_EMBEDDING_DIMENSION = 768
ITEM_KEYWORD_EMBEDDING_FIELD = 'question_vector'


def get_embedding(text, client, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        # print(response)
        return response.data[0].embedding  # Return the embedding vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def load_pqa(file_name, number_rows=1000):
    df = pd.DataFrame(columns=('question', 'answer'))
    with open(file_name) as f:
        i = 0
        for line in f:
            data = json.loads(line)
            df.loc[i] = [data['question_text'], data['answers'][0]['answer_text']]
            i += 1
            if (i == number_rows):
                break
    return df


def load_vectors(client: Redis, qa_list, vector_dict, vector_field_name, product_metadata):
    for index in product_metadata.keys():
        # Hash key
        key = 'product:' + str(index)

        # Hash values
        item_metadata = product_metadata[index]
        item_keywords_vector = vector_dict[index].astype(np.float32).tobytes()
        item_metadata[vector_field_name] = item_keywords_vector
        print("item_metadata", item_metadata)

        # HSET
        client.hset(key, mapping=item_metadata)


def create_hnsw_index(create_hnsw_index, vector_field_name, number_of_vectors, vector_dimensions=768, distance_metric='L2', M=40, EF=200):
    client_dev.ft(INDEX_NAME).create_index([
        VectorField("question_vector", "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions,
                    "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, }),
        TextField("question"),
        TextField("answer"),

    ])


def lambda_handler(event, context):
    # TODO implement

    qa_list = load_pqa('amazon-pqa/amazon_pqa_headsets.json', number_rows=1000)
    print("Loaded QA pairs")
    print(qa_list.head())
    product_metadata = qa_list.head(NUMBER_PRODUCTS).to_dict(orient='index')
    print("Loaded product metadata")
    item_keywords = [product_metadata[i]['question'] for i in product_metadata.keys()]
    item_keywords_vectors = [model.encode(sentence) for sentence in item_keywords]
    create_hnsw_index(client_dev, INDEX_NAME, NUMBER_PRODUCTS)
    print('Loading and Indexing + ' + str(NUMBER_PRODUCTS) + ' products')
    load_vectors(client_dev, product_metadata, item_keywords_vectors, ITEM_KEYWORD_EMBEDDING_FIELD, product_metadata)
    info = client_dev.ft(INDEX_NAME).info()
    num_docs = info['num_docs']
    space_usage = info['space_usage']
    num_indexed_vectors = info['num_indexed_vectors']
    vector_space_usage = (info['vector_space_usage'])

    print(f"{num_docs} documents ({space_usage} space used vectors indexed {
          num_indexed_vectors} vector space usage in {vector_space_usage}")
    topK = 5
    user_query = 'Does this work with xbox'
    # vectorize the query
    query_vector = model.encode(user_query).astype(np.float32).tobytes()
    # prepare the query
    q = Query(f'*=>[KNN {topK} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').paging(0,
                                                                                                    topK).return_fields('question', 'answer')
    params_dict = {"vec_param": query_vector}

# Execute the query
    results = client_dev.ft(INDEX_NAME).search(q, query_params=params_dict)
    # Print similar products and questions found
    for product in results.docs:
        print('***************Product  found ************')
        print('hash key = ' + product.id)
        print('question = ' + product.question)
        print('answer = ' + product.answer)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


if __name__ == "__main__":
    lambda_handler(None, None)



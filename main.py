from openai import OpenAI
import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from redis.commands.search.result import Result
import json
import pandas as pd


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


def lambda_handler(event, context):
    # TODO implement

    client_dev = redis.Redis(host='clustercfg.feedback-cluster.pu3xxp.memorydb.ap-south-1.amazonaws.com', port=6379,
                             decode_responses=True, ssl=True, ssl_cert_reqs="none"
                             )
    client_dev.ping()
    print("Connected to Redis")
    print("redis client", client_dev)
    qa_list = load_pqa('amazon-pqa/amazon_pqa_headsets.json', number_rows=1000)
    print("Loaded QA pairs")
    print(qa_list.head())

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


if __name__ == "__main__":
    lambda_handler(None, None)



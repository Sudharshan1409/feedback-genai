from openai import OpenAI
import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from redis.commands.search.result import Result
import json


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


def lambda_handler(event, context):
    # TODO implement

    client_dev = redis.Redis(host='clustercfg.feedback-cluster.pu3xxp.memorydb.ap-south-1.amazonaws.com', port=6379,
                             decode_responses=True, ssl=True, ssl_cert_reqs="none"
                             )
    client_dev.ping()

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


if __name__ == "__main__":
    lambda_handler(None, None)



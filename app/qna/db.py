import os
import redis
import numpy as np
import pandas as pd
import typing as t
import sklearn
import gensim
from sklearn.neighbors import NearestNeighbors
import openai
import re

from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.field import (
    VectorField,
    NumericField,
    TextField
)

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('qna/word_vectors.bin', binary=True)
question_embeddings = pd.read_csv('qna/question_embeddings.csv')
feedback_embeddings = pd.read_csv('qna/feedback_embeddings.csv')
question_docs = pd.read_csv('qna/question_dataframe.csv')
feedback_docs = pd.read_csv('qna/feedback_dataframe.csv')

openai.api_key = os.environ['OPENAI_API_KEY']

def get_questions(question_string):
    questions = question_string.strip().split("\n")
    result = []
    for question in questions:
        if any(char.isdigit() for char in question):
            result.append(question.strip())
    return result

# Returns array of questions

def api_get_question(job_description: str, experience_level: str, number_of_questions: int, content:str):
 

    prompt = f"Look for questions related to {job_description}, that most fit {experience_level} and put them in a list with this format:\nQuestion 1: \nQuestion 2:\nQuestion 3: \nFind the questions from the content below There should be {number_of_questions} questions:\n{content}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    questions = response.choices[0].text.strip()
    
    return get_questions(questions)

def api_get_feedback(question: str, user_response: str, job_description):
 

    question = 'Question 1: What are the different types of Machine Learning algorithms?'
    user_response = "knn and neural networks"
    job_description = 'Machine Learning'

    prompt = f"Act like you are giving feedback on a job interview, and are helping the person being interviewed improve. Based on this questions:  {question}\nAnd given this response: {user_response}\nFor this job: {job_description}\nGive constructive feedback for the response based on the content below. If you find the user's response to be a good answer the question, let them know and why. Otherwise, tell them how they could do better:\n[RELEVANT CONTENT]"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    feedback= response.choices[0].text.strip()
    print(feedback)

    
    return feedback

# feedback_docs =
# feedback_embeddings = 


INDEX_NAME = "embedding-index"
NUM_VECTORS = 4000
PREFIX = "embedding"
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"



def get_embeddings(text: str):
    df = pd.DataFrame(columns=['title', 'heading', 'content', 'tokens'])
    title = 'query'
    heading = 'query'
    content = text
    tokens = content.split()


    df.loc[len(df)] = [title, heading, content, tokens]

    # Generate embeddings for each row in the DataFrame
    embeddings = []
    for index, row in df.iterrows():
        text = row['content']  # the text column name in your CSV file
        words = text.split()
        vectors = []
        for word in words:
            try:
                vector = word_vectors[word]
                vectors.append(vector)
            except KeyError:
                # Ignore words that are not in the pre-trained word embeddings
                pass
        if vectors:
            # Calculate the mean of the word embeddings to get a document embedding
            doc_embedding = np.mean(vectors, axis=0)
            embeddings.append(doc_embedding)
        else:
            # Use a zero vector if none of the words are in the pre-trained word embeddings
            embeddings.append(np.zeros(100))

    # Add the embeddings as new columns in the DataFrame
    for i in range(100):
        df[i] = [embedding[i] for embedding in embeddings]

    # Save the DataFrame with the embeddings to a new CSV file
    return df

# Load documents and their embeddings
question_embeddings_df = question_embeddings.drop(columns=["title", "heading", "content"])
question_embeddings_arr = question_embeddings_df.to_numpy()
question_knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
question_knn_model.fit(question_embeddings_arr)

feedback_embeddings_df = feedback_embeddings.drop(columns=["title", "heading", "content"])
feedback_embeddings_arr = feedback_embeddings_df.to_numpy()
feedback_knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
feedback_knn_model.fit(feedback_embeddings_arr)



def get_most_relevant(is_questions: bool, text: str):
   
    # Load documents and their embeddings
    if is_questions:
        docs_df = question_docs 
    else:
        docs_df = feedback_docs
        
    # Load embedding of user query
    query_embedding = get_embeddings(text)
    query_embedding = query_embedding.drop(columns=["title", "heading", "content","tokens"]) # Drop the 'title' column
    query_embedding = query_embedding.to_numpy() # Convert to numpy array

    # Find the indices of the nearest neighbors to the query
    if is_questions:
        indices = question_knn_model.kneighbors(query_embedding, return_distance=False)
    else:
        indices = feedback_knn_model.kneighbors(query_embedding, return_distance=False)


    # Get the documents corresponding to the nearest neighbors
    top_5_knn_docs = docs_df.iloc[indices[0]]
    return top_5_knn_docs
    
def get_content_as_string(top_5: pd.DataFrame):

    top_5['content']
    content_string = '//\n'.join(top_5['content'])
    # Replace the newline characters ("\n") with a new line ("\n") and double slashes ("//")
    content_string = content_string.replace('\n', '\n')
    content_string = content_string[:3900]
    return (content_string)



def create_index(redis_conn: redis.Redis):
    # Define schema
    title = TextField(name="title")
    heading = TextField(name="heading")
    content = TextField(name="content")
    tokens = NumericField(name="tokens")
    embedding = VectorField("embedding",
        "FLAT", {
            "TYPE": "FLOAT64",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
            "INITIAL_CAP": NUM_VECTORS
        }
    )
    # Create index
    redis_conn.ft(INDEX_NAME).create_index(
        fields = [title, heading, content, tokens, embedding],
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

def process_doc(doc) -> dict:
    d = doc.__dict__
    if "vector_score" in d:
        d["vector_score"] = 1 - float(d["vector_score"])
    return d

def search_redis(
    redis_conn: redis.Redis,
    query_vector: t.List[float],
    return_fields: list = [],
    k: int = 5,
) -> t.List[dict]:
    """
    Perform KNN search in Redis.

    Args:
        query_vector (list<float>): List of floats for the embedding vector to use in the search.
        return_fields (list, optional): Fields to include in the response. Defaults to [].
        k (int, optional): Count of nearest neighbors to return. Defaults to 5.

    Returns:
        list<dict>: List of most similar documents.
    """
    # Prepare the Query
    base_query = f'*=>[KNN {k} @embedding $vector AS vector_score]'
    query = (
        Query(base_query)
         .sort_by("vector_score")
         .paging(0, k)
         .return_fields(*return_fields)
         .dialect(2)
    )
    params_dict = {"vector": np.array(query_vector, dtype=np.float64).tobytes()}
    # Vector Search in Redis
    results = redis_conn.ft(INDEX_NAME).search(query, params_dict)
    return [process_doc(doc) for doc in results.docs]

def list_docs(redis_conn: redis.Redis, k: int = NUM_VECTORS) -> pd.DataFrame:
    """
    List documents stored in Redis

    Args:
        k (int, optional): Number of results to fetch. Defaults to VECT_NUMBER.

    Returns:
        pd.DataFrame: Dataframe of results.
    """
    base_query = f'*'
    return_fields = ['title', 'heading', 'content']
    query = (
        Query(base_query)
        .paging(0, k)
        .return_fields(*return_fields)
        .dialect(2)
    )
    results = redis_conn.ft(INDEX_NAME).search(query)
    return [process_doc(doc) for doc in results.docs]

def index_documents(redis_conn: redis.Redis, embeddings_lookup: dict, documents: list):
    """
    Index a list of documents in RediSearch.

    Args:
        embeddings_lookup (dict): Doc embedding lookup dict.
        documents (list): List of docs to set in the index.
    """
    # Iterate through documents and store in Redis
    # NOTE: use async Redis client for even better throughput
    pipe = redis_conn.pipeline()
    for i, doc in enumerate(documents):
        key = f"{PREFIX}:{i}"
        embedding = embeddings_lookup[(doc["title"], doc["heading"])]
        doc["embedding"] = embedding.tobytes()
        pipe.hset(key, mapping = doc)
        if i % 150 == 0:
            pipe.execute()
    pipe.execute()

def load_documents(redis_conn: redis.Redis):
    # Load data
    docs = pd.read_csv("https://cdn.openai.com/API/examples/data/olympics_sections_text.csv")
    embeds = pd.read_csv("https://cdn.openai.com/API/examples/data/olympics_sections_document_embeddings.csv", header=0)
    max_dim = max([int(c) for c in embeds.columns if c != "title" and c != "heading"])
    embeds = {
           (r.title, r.heading): np.array([r[str(i)] for i in range(max_dim + 1)], dtype=np.float64) for _, r in embeds.iterrows()
    }
    print(f"Indexing {len(docs)} Documents")
    index_documents(
        redis_conn = redis_conn,
        embeddings_lookup = embeds,
        documents = docs.to_dict("records")
    )
    print("Redis Vector Index Created!")

def init():
    redis_conn = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=os.getenv('REDIS_PORT', 6379),
        password=os.getenv('REDIS_PASSWORD')
    )
    # Check index existence
    try:
        redis_conn.ft(INDEX_NAME).info()
        print("Index exists")
    except:
        print("Index does not exist")
        print("Creating embeddings index")
        # Create index
        create_index(redis_conn)
        load_documents(redis_conn)
    return redis_conn

def api_get_final_feedback(feedback, job_description):
 


    prompt = f"Act like you are giving feedback on a job interview, and are helping the person being interviewed improve. Based on all this feedback you gave:  {feedback}\nFor this job: {job_description}\nGive the person being inverviewd an overall score out of 100, then an overall summary on how they did."
    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

    final = response.choices[0].text.strip()

    # Extract the score from the final string
    score_match = re.search(r"\d+", final)
    score = int(score_match.group(0)) if score_match else None
    
    return final, score
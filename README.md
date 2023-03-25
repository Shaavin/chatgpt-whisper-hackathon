# Interview Preparation Using Redis & OpenAI

**Redis** plays a crucial role in the LLM & GenAI wave with it's ability to store, retrieve, and search with vector spaces in a low-latency, high-availability setting. With its heritage in enterprise caching, Redis has both the developer community and enterprise-readiness required to deploy quality AI-enabled applications in this demanding marketplace.

**OpenAI** is shaping the future of next-gen apps through it's release of powerful natural language and computer vision models that are used in a variety of downstream tasks.

This example Streamlit app gives you the tools to get up and running with **Redis** as a vector database and **OpenAI** as a LLM provider for embedding creation and text generation. _The combination of the two is where the magic lies._

## Run the App

Create your env file:

```bash
$ echo "OPENAI_API_KEY=<YOUR_OPENAI_KEY_GOES_HERE>
OPENAI_COMPLETIONS_ENGINE=text-davinci-003
OPENAI_EMBEDDINGS_ENGINE=text-embedding-ada-002
REDIS_HOST=redis
REDIS_PORT=6379
TOKENIZERS_PARALLELISM=false" > .env
```

_Fill out values, most importantly, your `OPENAI_API_KEY`._

Run with docker compose:

```bash
$ docker-compose up
```

_Add `-d` option to daemonize the processes to the background if you wish._

Navigate to:

```
http://localhost:8080/
```

The **first time you run the app** -- all documents will be downloaded, processed, and stored in Redis. This will take a few minutes to spin up initially. From that point forward, the app should be quicker to load.

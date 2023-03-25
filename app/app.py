import os
import streamlit as st

from urllib.error import URLError
from qna import answer_question_with_context
from dotenv import load_dotenv
load_dotenv()

try:
    ##################### Default page state data #####################
    default_job_description = ""
    default_resume_text = ""
    default_interview_type = "introductory"
    default_expertise_level ="Entry-level"
    default_interview_duration = 10
    default_is_interview_or_prompts = "interview"
    default_audio_response = ""
    default_interview_questions = []
    default_feedback = {
        'qualitative': "",
        'quantitative': -1,
    }

    #################### From the user to the model ####################
    # Setup questions to ask the user
    if "job_description" not in st.session_state:
        st.session_state['job_description'] = default_job_description
    if "resume_text" not in st.session_state:
        st.session_state['resume_text'] = default_resume_text
    if "interview_type" not in st.session_state:
        st.session_state['interview_type'] = default_interview_type # either "introductory" or "technical"
    if "expertise_level" not in st.session_state:
        st.session_state['expertise_level'] = default_expertise_level # "entry-level", "intermediate", or "expert"
    if "interview_duration" not in st.session_state:
        st.session_state['interview_duration'] = default_interview_duration
    if "is_interview_or_prompts" not in st.session_state:
        st.session_state['is_interview_or_prompts'] = default_is_interview_or_prompts # "interview" | "prompts"
    # Need Whisper AI to parse
    if "audio_response" not in st.session_state:
        st.session_state['audio_response'] = default_audio_response # !TODO: determine how this will work

    #################### From the model back to the user ####################
    if "interview_questions" not in st.session_state:
        st.session_state['interview_questions'] = default_interview_questions # string[]
    if "feedback" not in st.session_state:
        st.session_state['feedback'] = default_feedback

    st.write('Hello World')
    # st.image(os.path.join('assets','RedisOpenAI.png'))

    # col1, col2 = st.columns([4,2])
    # with col1:
    #     st.write("# Q&A Application")
    # with col2:
    #     with st.expander("Settings"):
    #         st.tokens_response = st.slider("Tokens response length", 100, 500, 400)
    #         st.temperature = st.slider("Temperature", 0.0, 1.0, 0.1)


    # question = st.text_input("*Ask thoughtful questions about the **2020 Summer Olympics***", default_question)

    # if question != '':
    #     if question != st.session_state['question']:
    #         st.session_state['question'] = question
    #         with st.spinner("OpenAI and Redis are working to answer your question..."):
    #             st.session_state['prompt'], st.session_state['response'] = answer_question_with_context(
    #                 question,
    #                 tokens_response=st.tokens_response,
    #                 temperature=st.temperature
    #             )
    #         st.write("### Response")
    #         st.write(f"Q: {question}")
    #         st.write(f"A: {st.session_state['response']['choices'][0]['text']}")
    #         with st.expander("Show Question and Answer Context"):
    #             st.text(st.session_state['prompt'])
    #     else:
    #         st.write(f"Q: {st.session_state['question']}")
    #         st.write(f"{st.session_state['response']['choices'][0]['text']}")
    #         with st.expander("Question and Answer Context"):
    #             st.text(st.session_state['prompt'].encode().decode())

    # st.markdown("____")
    # st.markdown("")
    # st.write("## How does it work?")
    # st.write("""
    #     The Q&A app exposes a dataset of wikipedia articles hosted by [OpenAI](https://openai.com/) (about the 2020 Summer Olympics). Ask questions like
    #     *"Which country won the most medals at the 2020 olympics?"* or *"Who won the men's high jump event?"*, and get answers!

    #     Everything is powered by OpenAI's embedding and generation APIs and [Redis](https://redis.com/redis-enterprise-cloud/overview/) as a vector database.

    #     There are 3 main steps:

    #     1. OpenAI's embedding service converts the input question into a query vector (embedding).
    #     2. Redis' vector search identifies relevant wiki articles in order to create a prompt.
    #     3. OpenAI's generative model answers the question given the prompt+context.

    #     See the reference architecture diagram below for more context.
    # """)

    # st.image(os.path.join('assets', 'RedisOpenAI-QnA-Architecture.drawio.png'))

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
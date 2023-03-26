# For recording audio
import os
from qna.db import get_content_as_string
from qna.db import get_most_relevant
import streamlit as st
from st_custom_components import st_audiorec
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
import time
import datetime
import math

from urllib.error import URLError
from qna import answer_question_with_context
from dotenv import load_dotenv
load_dotenv()

#################################################### Helper functions ####################################################
def parse_seconds_to_hrs_mins_secs(seconds):
    """
    Takes a number of seconds and converts it to hours, minutes, and seconds.
    These values are subtracted from the total as they are calculated, so each of them is not cumulative.
    """
    one_hour_in_secs = 60 * 60
    one_minute_in_secs = 60

    hours_elapsed = math.floor(seconds / one_hour_in_secs)
    seconds %= one_hour_in_secs
    mins_elapsed = math.floor(seconds / one_minute_in_secs)
    seconds %= one_minute_in_secs
    secs_elapsed = math.floor(seconds)

    return hours_elapsed, mins_elapsed, secs_elapsed

def format_num_to_double_digit_time(num):
    """
    Format numbers for time reporting. For example, the number 7 should be converted to "07", as seen in something like:
    00:07:14
    """
    return '00' if num == 0 else f'0{num}' if num < 10 else f'{num}'

def get_user_first_name():
    """
    Attempts to parse session state for the user's first name, else returns an empty string.
    """
    try:
        user_first_name = st.session_state.user_name.split()[0]
    except IndexError:
        user_first_name = ""
    return user_first_name

def check_can_progress_setup_state():
    """
    Logic checks for whether to enable the "NEXT PAGE" button during setup.
    """
    can_progress = st.session_state.user_name != "" and st.session_state.user_status != "" and st.session_state.desired_role != "" and st.session_state.user_company != ""
    st.session_state.can_progress_setup = can_progress

def check_can_progress_onboarding():
    """
    Logic checks for whether to enable the "START" button during onboarding.
    """
    can_progress = st.session_state.job_description != "" and bool(st.session_state.user_resume) and st.session_state.interview_focus != "" and st.session_state.num_interview_questions
    st.session_state.can_progress_onboarding = can_progress

def make_gap():
    """
    Make a large empty space on the page.
    """
    for i in range(10):
        st.markdown('')

def split_pages():
    """
    Separate pages with a large gap and a divider bar.
    """
    make_gap()
    st.markdown('___')
    make_gap()

def main():
    """
    Main page thread. This function renders the site.
    """
    MAXIMUM_NUM_INTERVIEW_QUESTIONS = 15 # !NOTE: Edit me to control the maximum number of interview questions allowed
  
    # Silently remove "user-audio.wav" if it exists
    # This file is used for tracking user audio
    for i in range(MAXIMUM_NUM_INTERVIEW_QUESTIONS):
        try:
            os.remove(f'user-answer-{i}.wav')
        except FileNotFoundError:
            pass

    st.session_state.update(st.session_state)

    # We need internet access to call the OpenAI APIs. If we don't have access, throw an error.
    try:
        ############################################# Default page state data #############################################

        default_user_name = ""
        default_user_status = ""
        default_desired_role = ""
        default_user_company = ""
        default_job_description = ""
        default_user_resume = ""
        default_interview_focus = ""
        default_num_interview_questions = 10 # How many the user wants, rather than the absolute maximum
        default_user_answers = [None] * MAXIMUM_NUM_INTERVIEW_QUESTIONS 
        default_interview_questions = [None] * MAXIMUM_NUM_INTERVIEW_QUESTIONS # !NOTE: Interview questions will not show for None values
        default_question_index = 0
        default_user_feedback = {
            "qualitative": [None] * MAXIMUM_NUM_INTERVIEW_QUESTIONS,
            "quantitative": -1, # !NOTE: Feedback will not display while quantitative feedback is equal to -1
        }
        default_can_progress_setup = False
        default_can_progress_onboarding = False

        ##################################### From the user, to be given to the model #####################################
        if 'user_name' not in st.session_state:
            st.session_state.user_name = default_user_name
        if 'user_status' not in st.session_state:
            st.session_state.user_status = default_user_status
        if 'desired_role' not in st.session_state:
            st.session_state.desired_role = default_desired_role
        if 'user_company' not in st.session_state:
            st.session_state.user_company = default_user_company
        if 'job_description' not in st.session_state:
            st.session_state.job_description = default_job_description
        if 'user_resume' not in st.session_state:
            st.session_state.user_resume = default_user_resume
        if 'interview_focus' not in st.session_state:
            st.session_state.interview_focus = default_interview_focus
        if 'num_interview_questions' not in st.session_state:
            st.session_state.num_interview_questions = default_num_interview_questions
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = default_user_answers

        ##################################### From thet model, to be given to the user #####################################
        # !NOTE: Edit me to see questions listed!
        if 'interview_questions' not in st.session_state:
            st.session_state.interview_questions = default_interview_questions
        # !NOTE: Edit me to see feedback in last page! Quantitative is overall, qualitative is for each (list)
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = default_user_feedback

        ######################################### General state that needs tracked #########################################
        if 'question_index' not in st.session_state:
            st.session_state.question_index = default_question_index
        if 'can_progress_setup' not in st.session_state:
            st.session_state.can_progress_setup = default_can_progress_setup
        if 'can_progress_onboarding' not in st.session_state:
            st.session_state.can_progress_onboarding = default_can_progress_onboarding

        ################################################ Set page meta-data ################################################
        st.set_page_config(
            page_title="Worktern Interview Prep"
        )

        ################################################### Page display ###################################################
        with st.container():
            st.markdown("### :blue[Welcome to Worktern's Interview AI]")
            st.text_input('', placeholder='First & Last Name', label_visibility='collapsed', on_change=check_can_progress_setup_state, key='user_name')
            st.selectbox('', ("I'm a...", 'High School Student', 'College Student', 'Professional'), label_visibility='collapsed', on_change=check_can_progress_setup_state, key='asfd')
            st.text_input('', placeholder="Job Title I'm Interviewing For...", label_visibility='collapsed', on_change=check_can_progress_setup_state, key='desired_role')
            st.text_input('', placeholder='Company Name', label_visibility='collapsed', on_change=check_can_progress_setup_state, key='user_company')
            st.button('NEXT PAGE', type='primary', disabled=(not st.session_state.can_progress_setup), use_container_width=True)

        split_pages()

        with st.container():
            st.markdown('### :blue[InterviewAI]')
            st.markdown('Ready to ace your interview? Make sure to upload all the info we need to customize your experience.')
            with st.expander('Insert Job Description'):
                st.text_area('Copy/paste:', placeholder='Place text here', key='job_description', on_change=(check_can_progress_onboarding))
            with st.expander('Upload Resume'):
                st.session_state.user_resume = st.file_uploader('Upload your file:', type=['pdf'], on_change=(check_can_progress_onboarding))
            st.session_state.interview_focus = st.selectbox('', ('Interview Type', 'Standard Interview (mixed)', 'Technical Interview (hard-skills)', 'Cultural Fit Interview (soft-skills)'), label_visibility='collapsed', on_change=(check_can_progress_onboarding))
            st.session_state.num_interview_questions = st.selectbox('', ('5 Questions', '10 Questions', '15 Questions'), label_visibility='collapsed', on_change=(check_can_progress_onboarding))
            st.button('START', type='primary', disabled=(not st.session_state.can_progress_onboarding), use_container_width=True)

        split_pages()

        with st.container():
            user_first_name = get_user_first_name() if get_user_first_name() != "" else "there"
            st.markdown(f'<p style="font-size: 18px;">Hi {user_first_name}, nice to meet you! My name is Interview AI, and I will be conducting your interview today.</p>', unsafe_allow_html=True)
            if st.session_state.interview_questions[st.session_state.question_index]:
                st.markdown(f'<p style="font-size: 18px;">{st.session_state.interview_questions[st.session_state.question_index]}</p>', unsafe_allow_html=True)
                st.markdown('___')
                st.write(f'Question {st.session_state.question_index + 1}')
                wav_audio_data = st_audiorec() # !NOTE: audio information stored in this variable as a byte array
                if st.button('Submit Answer', type='primary', disabled=(not wav_audio_data), use_container_width=True):
                    with open(f'user-answer-{st.session_state.question_index}.wav', 'bw') as file:
                        file.write(wav_audio_data)
                try:
                    audio_file = open(f'./user-answer-{st.session_state.question_index}.wav', 'rb')
                    transcript = openai.Audio.transcribe('whisper-1', audio_file)
                    st.session_state.user_answers[st.session_state.question_index] = transcript.text
                    st.session_state.question_index += 1
                except FileNotFoundError as e:
                    pass # Want to  fail silently, rather than noisely

        split_pages()

        if st.session_state.user_feedback['quantitative'] != -1:
            with st.container():
                user_first_name = get_user_first_name()
                st.markdown(f'<p style="text-align: center; font-weight: 700; font-size: 20px;">Great job{f", {user_first_name}" if user_first_name else ""}!</p>', unsafe_allow_html=True)
                st.markdown(f'''<p
                    style="display: grid; place-content: center; border-radius: 99999px; background-color: rgb(22 163 74); width: 75px; height: 75px; margin-left: auto; margin-right: auto; font-size: 24px"
                    >
                        {st.session_state.user_feedback["quantitative"]}
                    </p>''', unsafe_allow_html=True)
                st.markdown(f'<p style="font-weight: 700; text-align: center;">You scored {st.session_state.user_feedback["quantitative"]} out of 100</p>', unsafe_allow_html=True)
                st.markdown('Powered by ChatGPT & Whisper, Interview AI has given you a score based on the average results of all your interview questions! See the breakdown of your feedback & recommendations below...')
                for i in range(len(st.session_state.num_interview_questions)):
                    if st.session_state.user_answers[i]:
                        with st.expander(f'Question {i}'):
                            st.markdown(f'<p style="font-weight: 700; font-size: 18px;">{st.session_state.interview_questions[i]}</p>', unsafe_allow_html=True)
                            st.write(f'<p style="font-size: 14px;">{st.session_state.user_feedback["qualitative"][i]}</p>', unsafe_allow_html=True)
                            audio_file = open(f'user-answer-{i}.wav', 'rb')
                            st.audio(audio_file)

    ### TIMER LOGIC -- TODO (be careful with this; in its current state, it will block the main thread) ###
    # start_time = time.time()
    # stop_timer = False
    # timer_body = st.empty()
    # stop_timer_button = st.empty()
    # while not stop_timer:
    #     time_answering_question = round((time.time() - start_time), 2)
    #     hours, mins, secs = parse_seconds_to_hrs_mins_secs(time_answering_question)
    #     formatted_hours = format_num_to_double_digit_time(hours)
    #     formatted_mins = format_num_to_double_digit_time(mins)
    #     formatted_secs = format_num_to_double_digit_time(secs)
    #     timer_body.markdown(f"<p style='font-size: 26px; text-align: center; letter-spacing: 0.05em;'>{formatted_hours}:{formatted_mins}:{formatted_secs}</p>", unsafe_allow_html=True)
    #     time.sleep(1)
    #     stop_timer_button.button('Stop Recording', on_click=())


# input(header, input_type, input_label, key, options=()):

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
            """
            % e.reason
        )

main()
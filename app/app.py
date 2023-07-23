import streamlit as st
from striprtf.striprtf import rtf_to_text
import tempfile
import os
import re

uploaded_file = st.file_uploader("Choose a file")

def _replace_hyperlinks(text):
    # Define the regular expression pattern for hyperlinks
    HYPERLINKS = re.compile(r"(?:http|ftp|https)://\S+")

    # Define the replacement function for hyperlinks
    def _is_hyperlink(match):
        return f"[{match.group(0)}]({match.group(0)})"

    # Replace hyperlinks with Markdown-style links
    return re.sub(HYPERLINKS, _is_hyperlink, text)

if uploaded_file is not None:
    # Write the uploaded file to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Convert the temporary file to plain text using the "rtf_to_text" function from striprtf
    text = rtf_to_text(open(tmp_file_path, 'rb').read().decode('utf-8'))

    # Replace hyperlinks in the text using the "_replace_hyperlinks" function
    text = _replace_hyperlinks(text)
    # ///////////
    delimiters = ['documentation\n', 'decision\n', 'discussion\n']

    if 'Issues\n' in text:
        st.write("Converted text:")
        st.write(text)
        t =  text.split('Issues\n')[1].lower()
        temp_dict = {}
        for delimiter in delimiters:
            if delimiter in text:
                entry = text.split(delimiter)[0]
                temp_dict[len(entry)] = entry
                smallest_key = min(temp_dict.keys())
                Issues = temp_dict[smallest_key]
        else:
            Issues = text.split('\n\n')[0]
    else:
        st.write('Please Try Another Document')

        # ///////////
        

    # Display the converted text in the Streamlit app
    st.write("Converted text:")
    st.write(text)

    # Remove the temporary file from disk
    os.remove(tmp_file_path)

        

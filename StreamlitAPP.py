import os
import json
import traceback
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

# Load environment variables
load_dotenv()

# Load the JSON file for response
RESPONSE_JSON = None
with open('./config/workspace/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Create a title for the app
st.title("MCQs Creator Application with LangChain")

# Form for user inputs
with st.form("user_inputs"):
    # File upload field
    uploaded_file = st.file_uploader("Upload a PDF or TXT file")
    
    # Number of MCQs
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    
    # Subject input field
    subject = st.text_input("Insert Subject", max_chars=20)
    
    # Tone of questions
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
    
    # Submit button
    button = st.form_submit_button("Create MCQs")

# Check if the button is clicked and all fields have input
if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Loading..."):
        try:
            # Read content from uploaded file
            text = read_file(uploaded_file)
            
            # Count tokens and calculate cost of API call
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    text=text,
                    number=mcq_count,
                    subject=subject,
                    tone=tone,
                    response_json=json.dumps(RESPONSE_JSON)
                )
            
            # Print token details
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost: {cb.total_cost}")
            
            # Check if response is valid
            if isinstance(response, dict):
                # Extract quiz data from response
                quiz = response.get("quiz", None)
                if quiz is not None:
                    # Get table data and display as a DataFrame
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1  # Adjust index to start from 1
                        st.table(df)  # Display table
                        
                        # Display review in a text area
                        st.text_area(label="Review", value=response.get("review", ""))
                    else:
                        st.error("Error in processing the table data.")
                else:
                    st.error("No quiz data found in the response.")
            else:
                st.error("Invalid response format.")
        except Exception as e:
            # Log the error and display to user
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("An error occurred while processing your request.")
else:
    st.warning("Please fill out all the required fields and upload a file.")

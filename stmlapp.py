import streamlit as st
import openai
import pandas as pd
import os
import re
import dateparser
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API key from .env file
load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Load the roster data
roster_df = pd.read_csv("roster_cleaned.csv")
roster_df["Date"] = pd.to_datetime(roster_df["Date"], format="%d/%m/%Y", errors="coerce")

# Define shift columns
shift_columns = ["Day", "Night", "Day on call", "Night on call"]
WIFE_NAME = "Karunya Subramaniyan"

# Remove newlines and extra spaces from shift columns
for col in shift_columns:
    roster_df[col] = roster_df[col].astype(str).str.replace("\n", " ").str.strip()

### 📌 GPT CLASSIFIER FUNCTION
def classify_question(question):
    """Uses GPT to determine what type of shift info is needed."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that classifies shift-related questions strictly into categories:\n"
                                          "- 'next_shift'\n"
                                          "- 'next_day_shift'\n"
                                          "- 'next_night_shift'\n"
                                          "- 'next_day_on_call'\n"
                                          "- 'next_night_on_call'\n"
                                          "- 'colleagues_on_shift'\n"
                                          "- 'colleagues_on_date'\n"
                                          "- 'my_shift_on_date'\n"
                                          "- 'shift_run_end'\n"
                                          "Return ONLY one of these exact words. Do NOT generate any extra text."},
            {"role": "user", "content": f"User question: {question}\n\nReturn ONLY one category label."}
        ]
    )
    
    return response.choices[0].message.content.strip().lower()

### 📌 DATE EXTRACTION FUNCTION
def extract_date(question):
    """Extracts a date from the user question, allowing multiple formats, and assigns the correct month."""
    parsed_date = dateparser.parse(question, settings={'DATE_ORDER': 'DMY'})
    
    if not parsed_date:
        today = datetime.today()
        day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', question)
        if day_match:
            day = int(day_match.group(1))
            month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)', question, re.IGNORECASE)
            month = datetime.strptime(month_match.group(1), "%B").month if month_match else today.month
            year = today.year
            parsed_date = datetime(year, month, day)
        else:
            return None  # No valid date found

    return parsed_date.strftime("%d/%m/%Y")

### 📌 SHIFT QUERY FUNCTIONS
def get_next_shift(shift_type=None):
    """Finds the next shift of a specific type, or any shift if shift_type is None."""
    today = pd.Timestamp.today().normalize()
    if shift_type:
        future_shifts = roster_df[
            (roster_df["Date"] >= today) & 
            (roster_df[shift_type].str.contains(WIFE_NAME, na=False))
        ]
    else:
        future_shifts = roster_df[
            (roster_df["Date"] >= today) & 
            (roster_df[shift_columns].apply(lambda row: WIFE_NAME in row.values, axis=1))
        ]

    if future_shifts.empty:
        return f"Hey Karunya! 🌸 No upcoming {shift_type if shift_type else 'shifts'} found. Enjoy your time off! 🎉"

    next_shift = future_shifts.sort_values("Date").iloc[0]
    formatted_date = next_shift["Date"].strftime("%A, %d %B %Y")
    shift_info = shift_type if shift_type else ", ".join([col for col in shift_columns if next_shift[col] == WIFE_NAME])
    return (f"Hey Karunya! 😊 Your next **{shift_info}** shift is on **{formatted_date}**. "
            f"Take care and have a great day! 💙")

def get_my_shift_on_date(date_query):
    """Finds Karunya's shift on a specific date."""
    query_date = pd.to_datetime(date_query, format="%d/%m/%Y", errors="coerce")
    shift_row = roster_df[roster_df["Date"] == query_date]
    if shift_row.empty:
        return f"Hey Karunya! 😊 You have **no shifts** on **{query_date.strftime('%A, %d %B %Y')}**. Enjoy your day off! 🎉"
    formatted_date = query_date.strftime("%A, %d %B %Y")
    shifts = [col for col in shift_columns if shift_row.iloc[0][col] == WIFE_NAME]
    if not shifts:
        return f"Hey Karunya! 😊 You have **no shifts** on **{formatted_date}**. Enjoy your time off! 🎉"
    shift_details = ", ".join(shifts)
    return f"Hey Karunya! 😊 On **{formatted_date}**, you are working **{shift_details}**. Take care! 💙"

def get_colleagues_on_date(date_query):
    """Finds all colleagues working on a given date."""
    query_date = pd.to_datetime(date_query, format="%d/%m/%Y", errors="coerce")
    shift_row = roster_df[roster_df["Date"] == query_date]
    if shift_row.empty:
        return f"No shifts found on {query_date.strftime('%A, %d %B %Y')}."
    formatted_date = query_date.strftime("%A, %d %B %Y")
    shifts = {col: shift_row.iloc[0][col] for col in shift_columns if pd.notna(shift_row.iloc[0][col])}
    shift_details = "\n".join([f"  - {shift}: {name}" for shift, name in shifts.items()])
    return f"Here’s the shift schedule for **{formatted_date}**:\n\n{shift_details}\n\nHope this helps! 😊"

### 📌 MAIN FUNCTION: INTERPRET QUESTIONS
def ask_question(question):
    """Processes user queries dynamically using GPT classification."""
    category = classify_question(question)
    date_query = extract_date(question)

    # Define the mapping of categories to functions
    category_map = {
        "next_shift": lambda: get_next_shift(),
        "next_day_shift": lambda: get_next_shift("Day"),
        "next_night_shift": lambda: get_next_shift("Night"),
        "next_day_on_call": lambda: get_next_shift("Day on call"),
        "next_night_on_call": lambda: get_next_shift("Night on call"),
        "my_shift_on_date": lambda: get_my_shift_on_date(date_query),
        "colleagues_on_date": lambda: get_colleagues_on_date(date_query),
    }

    # Check if the category is valid
    if category in category_map:
        try:
            return category_map[category]()
        except Exception as e:
            return f"Oops! Something went wrong while processing your request. 😅 Please try again later. Error: {str(e)}"
    else:
        # If the category is not recognized, ask for clarification
        return ("Hmm, I didn't quite catch that. 🤔 Could you rephrase your question? "
                "For example, you can ask: 'When is my next shift?' or 'Who is working on 10/10/2023?'")

### 📌 STREAMLIT APP
st.title("👩‍⚕️ Roster Assistant for Karunya")
st.write("Ask me anything about your shifts!")

# Text input for user questions
user_question = st.text_input("Ask your question (or type 'exit' to quit):")

if user_question.lower() == "exit":
    st.write("Goodbye! Have a great day! 🌸")
else:
    if user_question:
        response = ask_question(user_question)
        st.write(f"🧑‍⚕️ Roster Assistant: {response}")
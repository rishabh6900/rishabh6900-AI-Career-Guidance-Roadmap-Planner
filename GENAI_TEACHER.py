import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(
    page_title="AI Career Guidance",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("AI Career Guidance & Roadmap Planner")
st.write(
    "Enter your interest area, education level, and dream career to get a personalized "
    "step-by-step guide with the right tech stack."
)

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

career_prompt = PromptTemplate(
    template=(
        "You are an expert career advisor.\n"
        "User's interest area: {interest}\n"
        "User's education level: {education}\n"
        "User's dream career: {dream_career}\n\n"
        "Generate a detailed, clear, step-by-step roadmap to achieve this career. "
        "Include the necessary skills, technologies (tech stack), certifications, "
        "and practical steps like projects or internships."
    ),
    input_variables=["interest", "education", "dream_career"],
)

# Chain the prompt to the model and output parser
career_chain = career_prompt | model | StrOutputParser()

with st.form("career_form"):
    interest = st.text_input("Your Interest Area (e.g., Data Science, Web Development, AI, Finance):")
    education = st.selectbox(
        "Your Education Level",
        ["High School", "Undergraduate", "Graduate", "Postgraduate", "Self-taught/Other"]
    )
    dream_career = st.text_input("Your Dream Career (e.g., Data Scientist, Software Engineer, Product Manager):")

    submitted = st.form_submit_button("Get Career Guidance")

if submitted:
    if not (interest.strip() and education.strip() and dream_career.strip()):
        st.warning("Please fill in all fields to get the guidance.")
    else:
        with st.spinner("Generating your personalized career roadmap..."):
            try:
                # Invoke Gemini model to get career plan
                output = career_chain.invoke({
                    "interest": interest,
                    "education": education,
                    "dream_career": dream_career
                })

                st.success("Here's your personalized career guidance:")
                st.markdown(output)

                # Extra styling for better readability
                st.markdown(
                    """
                    <style>
                    pre, p {
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your API key and try again.")

# Sidebar with instructions and tips
st.sidebar.header("How to use")
st.sidebar.write(
    """
    1. Enter your current interest or field you like.
    2. Select your highest education level.
    3. Specify your dream career.
    4. Click 'Get Career Guidance' to receive a personalized stepwise plan.
    
    This uses Gemini generative AI to create a tailored roadmap including skills, tools, and learning paths.
    """
)

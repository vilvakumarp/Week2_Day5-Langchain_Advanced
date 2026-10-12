
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Load API keys and LangSmith configuration
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Healthy Breakfast Generator",
    page_icon="🥗",
    layout="centered"
)

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Prompt template
PROMPT = """
Suggest exactly 5 healthy breakfast ideas.

Health criteria:
- Include protein-rich ingredients.
- Include fibre-rich ingredients such as vegetables,
  fruits, oats, or whole grains.
- Prefer minimally processed foods.
- Choose practical options for home cooking.

Instructions:
- Return exactly 5 numbered ideas.
- Give each idea a short description.
- Keep the answer clear and easy to read.
"""

# LangSmith tracing wrapper
@traceable(name="Healthy Breakfast Generator")
def generate_breakfast_ideas():
    reply = llm.invoke(PROMPT)
    return reply.content


# Streamlit UI
st.title("🥗 Healthy Breakfast Generator")

st.write(
    "Discover five healthy breakfast ideas "
    "powered by LangChain and OpenAI."
)

st.info(
    "The generator prioritizes protein, fibre, "
    "minimally processed ingredients, and practical meals."
)

if st.button("🍳 Generate Breakfast Ideas", type="primary"):
    with st.spinner("Generating your breakfast ideas..."):
        try:
            result = generate_breakfast_ideas()

            st.subheader("Your 5 Breakfast Ideas")
            st.markdown(result)

            st.success("Breakfast ideas generated successfully!")

        except Exception as e:
            st.error(
                "Unable to generate breakfast ideas. "
                "Check your API configuration and try again."
            )
            st.caption(f"Error details: {e}")

st.divider()

st.caption(
    "Built with Streamlit, LangChain, OpenAI and LangSmith tracing."
)

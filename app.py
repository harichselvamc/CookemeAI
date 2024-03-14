import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_hpCphWzchHdfgdfgdfgdeFHKdGOkllbf"
def download_model(model_url, model_path):
    # Check if the model file exists, if not, download it
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        os.system(f"wget {model_url} -O {model_path}")
        st.success("Model downloaded successfully!")
    else:
        st.info("Model already exists, skipping download...")

def cookmeai(user_input):
    try:
        model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin"
        model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"

        download_model(model_url, model_path)

        llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                "temperature": 0.01,
                "max_new_tokens": 256
            }
        )

        template = """ 
        you as master chef answer the user question {user_input} as summary about preparations steps in 5 line.
        """

        prompt = PromptTemplate(
            input_variables=["user_input"],
            template=template
        )

        response = llm.invoke(
            prompt.format(
                user_input=user_input,
            )
        )

        return response.split("\n")  # Split response into individual steps
    except Exception as e:
        return [f"An error occurred: {str(e)}"]

def main():
    # UI
    st.set_page_config(
        page_title="CookmeAI🍽️",
        page_icon="🍽️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.title("CookmeAI 🍽️")
    user_input = st.text_input("Enter your cooking question:")
    if st.button("AI Cook 🧑‍🍳"):
        if user_input:
            results = cookmeai(user_input)
            st.write("Master Advice:")
            for step in results:
                st.success(step)  # Display each step in a separate success box
                st.text("")  # Add a small gap between each step
        else:
            st.warning("Please enter a cooking question.")

if __name__ == "__main__":
    main()

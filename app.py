import streamlit as st
from src.ingest.ingest import ingest_data
from src.expert.answer import answer_question
import warnings
warnings.filterwarnings("ignore")
from src.utils.logger import get_logger
logger = get_logger(__name__)
def main():
    st.set_page_config(page_title="AI Mortgage Assistant", page_icon=":house:", layout="wide")
    st.title("AI Mortgage Assistant :house:")

    if st.button("Ingest Data"):
        with st.spinner("Ingesting data..."):
            ingest_data()
        st.success("Data ingested successfully!")
    
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Fetching answer..."):
            answer, context = answer_question(question)
        st.subheader("Answer:")
        st.write(answer)
        with st.expander("Context Used:"):
            for doc in context:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content)
                st.markdown("---")


if __name__ == "__main__":
    main()

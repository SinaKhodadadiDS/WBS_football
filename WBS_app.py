import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)


embedding_model = 'sentence-transformers/all-mpnet-base-v2'
embeddings_folder = 'C:\\WBS Coding School\\Projects\\Arsenal_Chatbot'

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

WBS_db = FAISS.load_local(r"C:\WBS Coding School\Projects\Arsenal_Chatbot", embeddings, allow_dangerous_deserialization=True)

retriever = WBS_db.as_retriever(search_kwargs={"k": 2})

memory = ConversationBufferMemory(memory_key = 'chat_history',
                                  return_messages = True,
                                  output_key = 'answer')  # Set output_key to 'answer'

template = """You are a nice chatbot having a conversation with a human. Answer the question. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""

prompt = PromptTemplate(template = template,
                        input_variables = ["context", "question"])

chain = ConversationalRetrievalChain.from_llm(llm,
                                                retriever = retriever,
                                                memory = memory,
                                                return_source_documents = False,
                                                combine_docs_chain_kwargs = {"prompt": prompt})
##### streamlit #####

st.title("Truth-telling bot")

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the rabbithole for answers..."):

        # send question to chain to get answer
        answer = chain(prompt)

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

import os, sys
import langchain.llms
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import time
from langchain.llms import OpenAI 
 

os.environ['REPLICATE_API_TOKEN'] = "r8_D7rrmEmSDn2b9q7o2d89Z3VqyVozYqV0gYGOy"
loader = PyPDFLoader('/workspaces/chatbot/salifou_cv.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=None)

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000})

# llm = Replicate(
#     model="dbmdz/bert-base-french-europeana-cased",
#     input={"temperature": 0.75, "max_length": 3000})


qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=False)

chat_history = []

while True:
    query = input('\n Question: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    
    result = qa_chain({'question': query, 'chat_history': chat_history})
    
    if result['answer']:
        response = result['answer']
        print("\nBot: \n")
        for char in response:
            print(char, end='', flush=True)
            time.sleep(0.05)  # Délai de 0.05 seconde entre chaque caractère
        print('\n')
        chat_history.append((query, result['answer']))
    else:
        print("I'm not sure how to respond to that.\n")
    
    if "merci" in query.lower():
        print("De rien !")
        chat_history.append(("De rien !", "You're welcome!"))
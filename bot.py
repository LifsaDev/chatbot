import os, sys
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import time, shelve
 
os.environ['REPLICATE_API_TOKEN'] = "r8_GCKbWwPSPIDmKa0dvzw5sdGfNL2A34U1EAXlV"
loader = PyPDFLoader('/workspaces/chatbot/salifou_cv.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
#vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="vectordb")
 
vectordb = Chroma(persist_directory="vectordb", embedding_function=embeddings)
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 3000})

 

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
        print(chat_history)
    else:
        print("I'm not sure how to respond to that.\n")
    
    if "merci" in query.lower():
        print("De rien !")
        chat_history.append(("De rien !", "You're welcome!"))
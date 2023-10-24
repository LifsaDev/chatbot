import os
import sys
import time
import json
from py3langid.langid import LanguageIdentifier, MODEL_FILE 
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from deep_translator import GoogleTranslator

os.environ['REPLICATE_API_TOKEN'] = "r8_GCKbWwPSPIDmKa0dvzw5sdGfNL2A34U1EAXlV"

def get_lang_source(text): 
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True) 
    lang, _ = identifier.classify(text) 
    return lang if lang in ["fr", "en"] else "en"

def xfile_to_text(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def process_multiple_pdfs(file_list):
    texts = []
    for file in file_list:
        texts.append(xfile_to_text(file))
    return texts

def save_chat_history(history, filename):
    with open(filename, 'w') as file:
        json.dump(history, file)

def load_chat_history(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as file:
            history = [tuple(item) for item in json.load(file)]
            return history
    else:
        return []

chat_history_filename = "chat_history.json"
chat_history = load_chat_history(chat_history_filename)

embeddings = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory="newdir", embedding_function=embeddings)


file_list = ['/workspaces/chatbot/story_telling.pdf','/workspaces/chatbot/Business_Analysis_FaceRecognition_&_ANPR.pdf']

# Liste des textes correspondants à chaque fichier PDF
texts_list = process_multiple_pdfs(file_list)
for text in texts_list:
    vectordb.add_documents(text)
vectordb.persist()

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 3000}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=False)

while True:
    query = input('\n Question: ')
    if query.lower() in ["exit", "quit", "q"]:
        save_chat_history(chat_history, chat_history_filename)
        print('Exiting')
        sys.exit()
    
    result = qa_chain({'question': query, 'chat_history': chat_history})
    
    if result['answer']:
        response = result['answer']
        
        # Détecter la langue de la question
        question_language = get_lang_source(query)
        response_language = get_lang_source(response)
        translated_response = GoogleTranslator(source=response_language, target=question_language).translate(response)
        print("\nBot: \n")
        for char in translated_response:
            print(char, end='', flush=True)
            time.sleep(0.05)
        print('\n')
        
        chat_history.append((query, result['answer']))
        save_chat_history(chat_history, chat_history_filename)
    else:
        print("I'm not sure how to respond to that.\n")


#from langchain.chat_models import init_chat_model
#from langchain_openai import OpenAIEmbeddings
#from langchain_core.vectorstores import InMemoryVectorStore
#import bs4
#from langchain import hub
#from langchain_community.document_loaders import WebBaseLoader
#from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from typing_extensions import List, TypedDict

from fastapi import FastAPI, File , UploadFile 
from typing import Annotated
import pymupdf


app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/ingest")
async def ingest_document():
    
    return {"Ingest": "Ingested"}

@app.post("/files")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload(file: Annotated[UploadFile, File()]):
    content = await file.read()
    
    doc = pymupdf.open(stream=content, filetype="pdf")
    out = open("output.txt", "wb")
    
    all_text = ""
    for page in doc:
        text = page.get_text()
        all_text += text
        out.write(text.encode("utf8"))
        out.write(bytes((12,)))
    
    out.close()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)    
    all_chunkin = text_splitter.split_text(text=all_text)
    print(all_chunkin)
    
    
    
    return {"filename": file.filename, "chunks": len(all_chunkin)}


'''
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        
llm = init_chat_model("gpt-4o-mini", model_provider="openai")        

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)


loader = WebBaseLoader(
    web_paths=("https://www.factored.ai/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("page_main")
        )
    ),
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Index chunks
document_ids = "ere"#vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "You are a assistente responsible to talk about companies" , 
     "question": "Tell me about Factored AI"}
).to_messages()

assert len(example_messages) == 1 
print (example_messages[0].content)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
'''


#response = graph.invoke({"question": "What is factored?"})
#print(response["answer"])
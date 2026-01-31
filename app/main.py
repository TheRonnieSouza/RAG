
#from langchain.chat_models import init_chat_model
#from langchain_openai import OpenAIEmbeddings
#from langchain_core.vectorstores import InMemoryVectorStore
#import bs4
#from langchain import hub
#from langchain_community.document_loaders import WebBaseLoader
#from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from typing_extensions import List, TypedDict
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from fastapi import FastAPI, File , UploadFile 
from typing import Annotated
import pymupdf
from uuid import uuid4
from app.config.settings import Settings

from langchain_core.documents import Document



app = FastAPI()

settings = Settings()

print(f"qdrant connection: {settings.qdrant.connection}")
client = QdrantClient(url=settings.qdrant.connection)

if not client.collection_exists("documents"):
    client.create_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE)
    )

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/ingest")
async def ingest_document():
    
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.google.embedding_model_name, api_key=settings.google.embedding_api_key)
    vector = embeddings.embed_query("hello, world!")
    
    vector_store = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings,
)
    document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    )
    document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
    metadata={"source": "news"},
    )
    document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    )
    document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    )
    document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    )
    document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    )
    document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    )
    document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    )  
    document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    )
    document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    )
    documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
        
    vector_store.add_documents(documents=documents, ids=uuids)
    
    vector[:5]
    print(f'vectores: {vector[0]}')
    
    return {vector[0],vector[1],vector[2], vector[3] }

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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=140,separators=['.'])    
    all_chunkin = text_splitter.split_text(text=all_text)
    
    all_chunking_saved = "" 
    chunck = open("chunk.txt", "wb")
    for chunkin in all_chunkin:
        all_chunking_saved += (chunkin + "\n\n")
        string =  (chunkin + "\n\n").encode("utf8")        
        chunck.write(string)
         
    print(all_chunkin)
    chunck.close()
    
    
    return {"filename": file.filename, "chunks": all_chunking_saved}


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
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langgraph.graph import StateGraph
from typing import Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from bs4 import BeautifulSoup

# 1. Cargar variables de entorno
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no está configurada") 

# 1. Cargar la página web
loader = RecursiveUrlLoader(
    url="https://www.promtior.ai/",
    max_depth=5,  # Profundidad máxima de navegación
    extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
    prevent_outside=True  # Solo URLs dentro del dominio
)
documents = loader.load()

# 2. Dividir el texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. Crear embeddings y vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. Crear el chain de conversación
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    return_generated_question=True, 
)

# 5. Configurar LangGraph para memoria
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

workflow = StateGraph(State)

def call_model(state: State):
    # Obtener la última pregunta
    question = state["messages"][-1].content
    # Obtener historial previo
    chat_history = [(msg.content, "") for msg in state["messages"][:-1]]
    
    # Consultar usando el chain que incluye la búsqueda en la web
    response = chain({
        "question": question,
        "chat_history": chat_history
    })
    parser = StrOutputParser()
    parsed_response = parser.parse(response["answer"])
    return parsed_response

# Configurar el grafo
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

# Compilar con memoria
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="Chatbot para web de Promtior",
)

# 8. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8003)
# adapted from https://github.com/mmz-001/knowledge_gpt

from functools import lru_cache
import re, os, dotenv
from io import BytesIO
from typing import Any, Dict, List, Optional

import docx2txt
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI, BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings


def parse_file(file: str) -> List[str]:
    """Parses a file and returns a list of strings, one for each page."""
    _, ext = os.path.splitext(file)
    with open(file, "rb") as f:
        if ext == ".pdf":
            return parse_pdf(f)
        elif ext == ".docx":
            return [parse_docx(f)]
        elif ext == ".txt":
            return [parse_txt(f)]
        else:
            raise ValueError(f"Unknown file extension: {ext}")

def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def text_to_docs(text: str | List[str], name: Optional[str] = None) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
        if name:
            doc.metadata["name"] = name

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            if name:
                metadata={"name":name, "page": doc.metadata["page"], "chunk": i}
            else:
                metadata={"page": doc.metadata["page"], "chunk": i}
            doc = Document(
                page_content=chunk, metadata=metadata
            )
            # Add sources a metadata
            if name:
                doc.metadata["source"] = f"{doc.metadata['name']}-{doc.metadata['page']}-{doc.metadata['chunk']}"
            else:
                doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

def load_env():
    """Loads the environment variables from a .env file"""
    dotenv.load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("No OpenAI API key found. Please add it to a .env file.")

def embed_docs(docs: List[Document], openai_embeddings=True, persist_directory=None) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""
    embeddings = OpenAIEmbeddings() if openai_embeddings else HuggingFaceHubEmbeddings()
    index = FAISS.from_documents(docs, embeddings)
    return index

        

def load_store(save_dir : str, openai_embeddings=True) -> VectorStore:
    """Loads index from a save directory"""
    embeddings = OpenAIEmbeddings() if openai_embeddings else HuggingFaceHubEmbeddings()
    index = FAISS.load_local(save_dir, embeddings)
    return index

def update_store(index: VectorStore, docs: List[Document]) -> bool:
    # avoid re-embedding existing documents
    existing_docs = {doc.metadata['name']  for doc in index.docstore._dict.values()}
    docs = [doc for doc in docs if doc.metadata['name'] not in existing_docs]
    if docs:
        index.add_documents(docs)
        return True
    else:    
        return False

def save_store(index: VectorStore, save_dir: str):
    """Saves a FAISS index to a save directory"""
    index.save_local(save_dir)

@lru_cache()
def search_docs(index: VectorStore, query: str, k:int = 5) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""
    # Search for similar chunks
    docs = index.similarity_search(query, k=k)
    return docs

@lru_cache()
def get_llm(temperature:float=0.0, model_name= "gpt-3.5-turbo") -> BaseLLM:
    """Gets an OpenAI LLM model."""
    load_env()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI if model_name == "gpt-3.5-turbo" else OpenAI
    llm = model(
        temperature=temperature, openai_api_key=OPENAI_API_KEY, model_name=model_name,
        streaming=True, callbacks=[StreamingStdOutCallbackHandler()]
    )  # type: ignore
    return llm

@lru_cache()
def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])

@lru_cache()
def create_template(template: str, input_variables: List[str] =["summaries", "question"]) -> PromptTemplate:
    return PromptTemplate(
        template=template, input_variables=["summaries", "question"])


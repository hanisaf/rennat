from functools import reduce
from io import BytesIO
from typing import List, Optional
from bs4 import BeautifulSoup
import os, dotenv, re
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt
from pypdf import PdfReader
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class Util:
    @staticmethod
    def parse_file(file: str) -> List[str]:
        """Parses a file and returns a list of strings, one for each page."""
        _, ext = os.path.splitext(file)
        with open(file, "rb") as f:
            if ext.lower() == ".pdf":
                return Util.parse_pdf(f)
            elif ext.lower() == ".docx":
                return [Util.parse_docx(f)]
            elif ext.lower() == ".txt":
                return [Util.parse_txt(f)]
            elif ext.lower() == ".html":
                return [Util.parse_html(f)]
            else:
                raise ValueError(f"Unknown file extension: {ext}")

    @staticmethod
    def parse_html(file: BytesIO) -> str:
        # using BeautifulSoup to parse the HTML
        soup = BeautifulSoup(file, 'html.parser')
        # get the text out of the soup
        text = soup.get_text()
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
    def parse_docx(file: BytesIO) -> str:
        text = docx2txt.process(file)
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
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

    @staticmethod
    def parse_txt(file: BytesIO) -> str:
        text = file.read().decode("utf-8")
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
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

    @staticmethod
    def text_match(text1: str, text2: str, how: str = "and"):
        tokens1 = re.split(r"\W+", text1.lower())
        tokens2 = re.split(r"\W+", text2.lower())
        included = [t in tokens2 for t in tokens1 if len(t) > 1]
        if how == "and":
            return reduce(lambda x, y: x and y, included, True)
        elif how == "or":
            return reduce(lambda x, y: x or y, included, False)
        else:
            raise ValueError(f"Unknown how: {how}")
        
    @staticmethod
    def seek_openai_token():
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            dotenv.load_dotenv()
            token = os.getenv("OPENAI_API_KEY")
        return token
        
    @staticmethod
    def total_words(docs: List[Document]) -> int:
        words = 0
        for doc in docs:
            words += len(re.split("\W", doc.page_content)) 
            words += len(re.split("\W", doc.metadata["source"]))
        return words + 2
    
    @staticmethod
    def estimated_tokens(docs: List[Document]) -> int:
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them 
        return Util.total_words(docs) * 2
    
    @staticmethod
    def max_sources(docs: List[Document], max_tokens = 4096) -> int:
        for i in range(len(docs)):
            if Util.estimated_tokens(docs[:i]) > max_tokens:
                break
        return i - 1

    @staticmethod
    def get_llm(openai_token: str, temperature:float=0.0, model_name= "gpt-3.5-turbo", streaming=True) -> BaseLLM:
        """Gets an OpenAI LLM model."""
        if not openai_token:
            openai_token = Util.seek_openai_token()
        model = ChatOpenAI if model_name == "gpt-3.5-turbo" else OpenAI
        llm = model(
            temperature=temperature, openai_api_key=openai_token, model_name=model_name,
            streaming=streaming, callbacks=[StreamingStdOutCallbackHandler()]
        )  
        return llm
    
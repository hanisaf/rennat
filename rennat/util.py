from functools import lru_cache, reduce
from io import BytesIO
from typing import List, Optional
from bs4 import BeautifulSoup
import os, dotenv, re
from langchain import LLMChain, PromptTemplate
from langchain.llms import GPT4All
from langchain.llms import LlamaCpp
from langchain import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt
from pypdf import PdfReader
import ebooklib
from ebooklib import epub

#from PyPDF4 import PdfFileReader
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from chromadb.api.types import GetResult
class Util:
    MAX_PROMPT_TOKENS = 4096

    @staticmethod
    def parse_file(file_name: str) -> List[str]:
        """Parses a file and returns a list of strings, one for each page."""
        _, ext = os.path.splitext(file_name)
        with open(file_name, "rb") as file_content:
            if ext.lower() == ".pdf":
                return Util.parse_pdf(file_content, file_name)
            elif ext.lower() == ".docx":
                return [Util.parse_docx(file_content, file_name)]
            elif ext.lower() == ".txt":
                return [Util.parse_txt(file_content, file_name)]
            elif ext.lower() == ".html":
                return [Util.parse_html(file_content, file_name)]
            elif ext.lower() == ".epub":
                return Util.parse_epub(file_content, file_name)
            else:
                raise ValueError(f"Unknown file extension: {ext}")

    @staticmethod
    def parse_html(file: BytesIO, name: str = None) -> str:
        # using BeautifulSoup to parse the HTML
        soup = BeautifulSoup(file, 'html.parser')
        # get the text out of the soup
        text = soup.get_text()
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
    def parse_docx(file: BytesIO, name: str = None) -> str:
        text = docx2txt.process(file)
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
    def parse_pdf(file: BytesIO, name: str = None) -> List[str]:
        pdf = PdfReader(file, strict = False)
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
    def parse_txt(file: BytesIO, name: str = None) -> str:
        text = file.read().decode("utf-8")
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text

    @staticmethod
    def parse_epub(file: BytesIO, name: str = None) -> List[str]:
        """Uses ebooklib to parse an epub file and returns a string.
           Returns a list of strings, one for each chapter.""" 
        book = epub.read_epub(name)
        output = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    # get the text out of the soup
                    text = soup.get_text()
                    # Remove multiple newlines
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    output.append(text)
                except Exception as e:
                    print("Error parsing epub part " + item.get_name(), e)
        return output

        pass
    
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
    def max_sources(docs: List[Document], max_tokens = MAX_PROMPT_TOKENS) -> int:
        for i in range(len(docs)):
            if Util.estimated_tokens(docs[:i]) > max_tokens:
                return i - 1
        return len(docs)

    @staticmethod 
    @lru_cache(maxsize=32)
    def get_llm(model_token_or_path: str, model_name= "gpt-3.5-turbo", temperature:float=0.0, streaming=True) -> BaseLLM:
        """Gets an OpenAI LLM model."""
        if not model_token_or_path:
            model_token_or_path = Util.seek_openai_token()
        
        callbacks=[StreamingStdOutCallbackHandler()]
        if model_name == "gpt-3.5-turbo":
            llm = ChatOpenAI(
                temperature=temperature, openai_api_key=model_token_or_path, model_name=model_name,
                streaming=streaming, callbacks=callbacks
            )
        elif model_name == "text-davinci-003":
            llm = OpenAI(
                temperature=temperature, openai_api_key=model_token_or_path, model_name=model_name,
                streaming=streaming, callbacks=callbacks
            )
        elif model_name == "huggingface":      
            repo_id = "Writer/camel-5b-hf" # See https://huggingface.co/Writer for other options
            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64}, huggingfacehub_api_token=model_token_or_path)
        elif model_name == "gpt4all":
            llm = LlamaCpp(model_path=model_token_or_path, callbacks=callbacks, verbose=True)   
        return llm
    
    @staticmethod
    def getresult2listdoc(results: GetResult) -> List[Document]:
        """convert between chromadb and langchain format of results"""
        ids = results['ids']
        metadatas = results['metadatas']
        texts = results['documents']
        docs = [Document(
                    page_content=texts[i], metadata=metadatas[i], id=ids[i]
                ) for i in range(len(results['ids']))]
        return docs 
    
    @staticmethod
    def gen_citation(doc: Document) -> str:
        author = doc.metadata['name'].split(' - ')[0]
        page = doc.metadata['page']
        citation = f"({author}, p. {page})"
        return citation
    
if __name__ == "__main__":
    # testing code
    model_name = input("Enter model name: ")
    model_token = input("Enter model token: ")
    llm = Util.get_llm(model_token_or_path=model_token, model_name=model_name)

    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "Who won the FIFA World Cup in the year 1994? "

    print(llm_chain.run(question))    
    pass
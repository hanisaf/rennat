from functools import lru_cache, reduce
from io import BytesIO
from typing import List, Optional
import uuid, os, re
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt
from pypdf import PdfReader
from tqdm import tqdm


class Index:
    def __init__(self, index_file: str = None, collection_name: str = "mycollection") -> None:
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=index_file
        ))

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, docs: List[Document]) -> None:
        documents = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [str(uuid.uuid1()) for _ in docs]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        self.client.persist()

    def add_files(self, files: List[str], verbose: bool = False, ignore_existing: bool = True) -> None:
        if ignore_existing:
            existing_files = self.get_file_names()
            ignored_files = [f for f in files if os.path.basename(f) in existing_files]
            print("Ignoring ", len(ignored_files), " existing files.")            
            files = [f for f in files if os.path.basename(f) not in existing_files]
        docs = []
        for f in tqdm(files):
            try:
                text = Index.parse_file(f)
                fn = os.path.basename(f)
                if verbose:
                    print("Processing ", fn, "...")
                doc = Index.text_to_docs(text, fn)
                docs.extend(doc)
            except Exception as e:
                print("Error processing ", fn, ": ", e)
        if docs: # not empty
            self.add_documents(docs)

    def search_docs(self, query: str, k:int = 5, meta_names : List[str] = None, exclude_names : List[str] = None, min_length=0) -> List[Document]:
        """Searches index for similar chunks to the query
        and returns a list of Documents."""
        # Search for similar chunks
        n=k
        # broad search then narrow down
        if min_length or meta_names or exclude_names:
            k = self.collection.count()

        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        # assemble a list of documents from results
        ids = results['ids']
        metadatas = results['metadatas']
        texts = results['documents']
        docs = [Document(
                    page_content=texts[i], metadata=metadatas[i], id=ids[i]
                ) for i in range(len(results))]
        # filter out documents that don't meet the criteria        
        if meta_names:
            docs = [doc for doc in docs if doc.metadata['name'] in meta_names]
        if exclude_names:
            docs = [doc for doc in docs if doc.metadata['name'] not in exclude_names]
        if min_length:
            docs = [doc for doc in docs if len(doc.page_content) > min_length]
        return docs[:n]

    @lru_cache()
    def get_file_names(self) -> List[str]:
        """Gets a list of file names."""
        docstore = self.collection.get()
        existing_docs = {d['name'] for d in docstore['metadatas']}
        return existing_docs

    def search_meta(self, query: str | List[str], how: str = "and") -> List[Document]:
        """Search based on the name of the document."""
        names = self.get_doc_names()
        # if query is a string perform a single search
        if isinstance(query, str):
            matches = [name for name in names if Index.text_match(query, name, how=how)]
        # if query is a list of strings perform multiple searches
        elif isinstance(query, list):
            matches = []
            for q in query:
                matches += [name for name in names if Index.text_match(q, name, how=how)]
        return matches
    
    def get_langchain_chroma(self) -> Chroma:
        return Chroma(client = self.client, collection_name=self.collection_name)

    @staticmethod
    def parse_file(file: str) -> List[str]:
        """Parses a file and returns a list of strings, one for each page."""
        _, ext = os.path.splitext(file)
        with open(file, "rb") as f:
            if ext == ".pdf":
                return Index.parse_pdf(f)
            elif ext == ".docx":
                return [Index.parse_docx(f)]
            elif ext == ".txt":
                return [Index.parse_txt(f)]
            elif ext == ".html":
                return [Index.parse_html(f)]
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
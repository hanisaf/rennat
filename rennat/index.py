from functools import reduce
from io import BytesIO
from typing import List, Optional
import uuid, os, re
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
import dotenv
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt
from pypdf import PdfReader
from tqdm import tqdm
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from util import Util
class Index:
    def __init__(self, index_file: str = None, collection_name: str = "references", openai_token:str = None) -> None:
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=index_file,
            anonymized_telemetry=False,
        ))
        self.switch_collection(collection_name)
        if not openai_token:
            self.openai_token = Util.seek_openai_token()
        else:
            self.openai_token = openai_token
        self.llm = None

    def switch_collection(self, collection_name: str) -> None:
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
                text = Util.parse_file(f)
                fn = os.path.basename(f)
                if verbose:
                    print("Processing ", fn, "...")
                doc = Util.text_to_docs(text, fn)
                docs.extend(doc)
            except Exception as e:
                print("Error processing ", fn, ": ", e)
        if docs: # not empty
            self.add_documents(docs)

    def size(self):
        return self.collection.count()


    def answer(self, query: str, sources: List[Document] = None, modifier="", verbose=False):
        if not self.llm:
            self.llm = Util.get_llm(self.openai_token)
        if not sources:
            sources = self.search_docs(query)
        
        description = "a" if not modifier else "an " + modifier if modifier[0] in "aeiou" else "a " + modifier
        prompt = Prompts.BASE_PROMPT.replace("/description/", description)
        prompt_words = len(re.split("\W", prompt))
        remaining_tokens = 4096 - prompt_words * 2
        i = Util.max_sources(sources, remaining_tokens)
        sources = sources[:i] # -2 to provide some fodder for chat

        prompt_template = Prompts.template_from_str(prompt)
        chain = load_qa_with_sources_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt_template,
            verbose=verbose)      

        answer = chain(
            {"input_documents": sources, "question": query}, return_only_outputs=False
        )

        text = answer["output_text"].strip()

        papers = []
        i = 0
        for doc in sources:
            if not doc.metadata['name'] in papers:
                papers.append(doc.metadata['name'])

        return text, papers, sources

    def search_docs(self, query: str, k:int = 1000, meta_names : List[str] = None, exclude_names : List[str] = None, min_length=0) -> List[Document]:
        """Searches index for similar chunks to the query
        and returns a list of Documents."""


        k = min(k, self.size())

        #TODO investigate why this is not working
        # results = self.collection.query(
        #     query_texts=[query],
        #     n_results=k - 1
        # )
        # # assemble a list of documents from results
        # ids = results['ids'][0] # only one query text
        # metadatas = results['metadatas'][0] # only one query text
        # texts = results['documents'][0] # only one query text
        # docs = [Document(
        #             page_content=texts[i], metadata=metadatas[i], id=ids[i]
        #         ) for i in range(len(results))]
                
        chroma = self.get_langchain_chroma()
        # create a filter for the metadata
        # and change k in case of a filter given much fewer results
        if meta_names:
            if len(meta_names) > 1:
                meta_or = [{
                    "name": {
                        "$eq": name
                    }
                } for name in meta_names] 
                filter = {"$or" : meta_or} 
            else:
                filter = {"name": {"$eq": meta_names[0]}}
        else:
            filter = None
        
        if filter:
            relevant_docs = self.collection.get(where=filter)
            k = len(relevant_docs['ids'])
        docs = chroma.similarity_search(query, k=k, filter=filter)


        # filter out documents that don't meet the criteria        
        if meta_names:
            docs = [doc for doc in docs if doc.metadata['name'] in meta_names]
        if exclude_names:
            docs = [doc for doc in docs if doc.metadata['name'] not in exclude_names]
        if min_length:
            docs = [doc for doc in docs if len(doc.page_content) > min_length]
        return docs

    def get_file_names(self) -> List[str]:
        """Gets a list of file names."""
        docstore = self.collection.get()
        existing_docs = {d['name'] for d in docstore['metadatas']}
        return existing_docs

    def search_meta(self, query: str | List[str], how: str = "and") -> List[Document]:
        """Search based on the name of the document."""
        names = self.get_file_names()
        # if query is a string perform a single search
        if isinstance(query, str):
            matches = [name for name in names if Util.text_match(query, name, how=how)]
        # if query is a list of strings perform multiple searches
        elif isinstance(query, list):
            matches = []
            for q in query:
                matches += [name for name in names if Util.text_match(q, name, how=how)]
        return matches
    
    def get_langchain_chroma(self) -> Chroma:
        return Chroma(client = self.client, collection_name=self.collection_name)


class Prompts:
    BASE_PROMPT = """Create /description/ final answer to the given questions using the provided sources.
QUESTION: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:"""
    @staticmethod
    def template_from_str(template: str, input_variables: List[str] =["summaries", "question"]) -> PromptTemplate:
        return PromptTemplate(
            template=template, input_variables=input_variables)
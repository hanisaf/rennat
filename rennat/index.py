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
from langchain.chains.summarize import load_summarize_chain

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
        
        # testing with a local model
        #self.llm = Util.get_llm(f"{os.path.expanduser('~')}/Library/Application Support/nomic.ai/GPT4All/ggml-stable-vicuna-13B.q4_2.bin", "gpt4all")

    def switch_collection(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def delete_collection(self, collection_name) -> None:
        if collection_name == "references":
            print("Cannot delete references collection.")
            return
        else:
            self.switch_collection("references")
            self.client.delete_collection(collection_name)
            self.client.persist()

    def list_collections(self) -> List[str]:
        return self.client.list_collections()
    
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
        for f in tqdm(files):
            try:
                fn = os.path.basename(f)
                text = Util.parse_file(f)
                if verbose:
                    print("Processing ", fn, "...")
                doc = Util.text_to_docs(text, fn)
                if doc:
                    self.add_documents(doc)
            except Exception as e:
                print("Error processing ", fn, ": ", e)
        self.client.persist()

    def delete_files(self, files: List[str]) -> None:
        # # this is generating an error when the file name has special characters
        # # it seems it depends on sql injection        
        # for f in tqdm(files):
        #     self.collection.delete(where={"name": {"$eq": f}})
        # print("deleted ", len(files), " files")
        # this is a workaround but it's memory intensive
        results = self.collection.get()
        ids = results['ids']
        metadatas = results['metadatas']
        deleted_ids = [ ids[i] for i in tqdm(range(len(ids))) if metadatas[i]['name'] in files ]
        self.collection.delete(ids=deleted_ids)
        self.client.persist()
        print("deleted ", len(files), " files with", len(deleted_ids), " chunks.")


    def size(self):
        return self.collection.count()

    def synthesize(self,query: str, sources: List[Document], verbose=False):
        if not self.llm:
            self.llm = Util.get_llm(self.openai_token)
        if not sources:
            sources = self.search_docs(query, k=5)
        

        for source in sources:
            citation = Util.gen_citation(source)
            source.page_content = f"Source citation: {citation}: " + source.page_content


        map_prompt = PromptTemplate(template=Prompts.SUMMARY_PROMPT, input_variables=["text"])
        combine_prompt = PromptTemplate(template=Prompts.COMBINE_PROMPT, input_variables=["text"])



        chain = load_summarize_chain(self.llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, verbose=verbose)
        answer = chain(
            {"input_documents": sources}, return_only_outputs=False
        )
        text = answer["output_text"].strip()
        papers = []
        i = 0
        for doc in sources:
            if not doc.metadata['name'] in papers:
                papers.append(doc.metadata['name'])        
        return text, papers, sources        

    def refine(self,query: str, sources: List[Document], verbose=False):
        if not self.llm:
            self.llm = Util.get_llm(self.openai_token)
        if not sources:
            sources = self.search_docs(query, k=10)
        
        #TODO: Original answer remains unchanged. Need to fix this.

        CHAT_REFINE_PROMPT_TMPL = (
            "The original question is as follows: {question}\n"
            "We have provided an existing answer, including sources: {existing_answer}\n"
            "We have the opportunity to refine the existing answer"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_str}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question. "
            "If you do update it, please update the sources as well. "
            "If the context isn't useful, return the original answer."
            "If the original answer remains unchanged, repeat the original answer."
        )
        CHAT_REFINE_PROMPT = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=CHAT_REFINE_PROMPT_TMPL,
        )

        chain = load_qa_with_sources_chain(self.llm, chain_type="refine", 
                            refine_prompt=CHAT_REFINE_PROMPT, verbose=verbose)


        answer = chain(
            {"input_documents": sources, "question": query}, return_only_outputs=False
        )
        text = answer["output_text"].strip()
        return text

    def answer(self, query: str, sources: List[Document] = None, modifier="", verbose=False):
        if not self.llm:
            self.llm = Util.get_llm(self.openai_token)
        if not sources:
            sources = self.search_docs(query)
        
        description = "a" if not modifier else "an " + modifier if modifier[0] in "aeiou" else "a " + modifier
        prompt = Prompts.BASE_PROMPT.replace("/description/", description)
        prompt_words = len(re.split("\W", prompt))
        remaining_tokens = Util.MAX_PROMPT_TOKENS - prompt_words * 2
        i = Util.max_sources(sources, remaining_tokens)
        sources = sources[:i] 

        prompt_template = PromptTemplate(template=prompt, input_variables=["summaries", "question"])

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

    def search_docs(self, query: str, k:int = 1000, meta_names : List[str] = None, exclude_names : List[str] = None, min_length=100) -> List[Document]:
        """Searches index for similar chunks to the query
        and returns a list of Documents."""
                
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
        else:
            k = min(k, self.size())

        results = self.collection.query(
            query_texts=[query],
            where=filter,
            n_results=k
        )
        # assemble a list of documents from results
        ids = results['ids'][0] # only one query text
        metadatas = results['metadatas'][0] # only one query text
        texts = results['documents'][0] # only one query text
        docs = [Document(
                    page_content=texts[i], metadata=metadatas[i], id=ids[i]
                ) for i in range(len(ids))]

        # filter out documents that don't meet the criteria        
        # if meta_names:
        #     docs = [doc for doc in docs if doc.metadata['name'] in meta_names]
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

    def get_documents(self, file_names: List[str]) -> List[Document]:
        docstore = self.collection.get()
        docs = Util.getresult2listdoc(docstore)
        if file_names:
            docs = [doc for doc in docs if doc.metadata['name'] in file_names]
        #TODO sort?
        return docs
    
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
    BASE_PROMPT = """Create /description/ final answer to the given question using the provided sources.
QUESTION: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:"""

    SUMMARY_PROMPT = """Write a concise summary of the following and put the source citation at the end:


"{text}"

CONCISE SUMMARY:
"""

    COMBINE_PROMPT = """Write a concise summary of the following and include all source citations verbatim:


"{text}"

CONCISE SUMMARY:
"""

"Source citation: (Stendal, Thapa, Lanamaki, p. 8):Given actors, have  the capabilities to perform the action. Likewise, we  found another point of confusion in the identification,  for example how and who identify the affordances, likewise when do the affordances identified.   The question of affordances as possibility of  action or affordance as performed action remains  unclear. The literatures reported in this paper have  reported list of affordan ces, but also insisted on  understanding the actualization and consequences of  identified affordances.   Finally, we agree that affordance is a useful lens  to understand the sociotechnical mechanism in  Information System context, though it needs critical  construction to progress towards maturity."
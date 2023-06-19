#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
import warnings
import os, sys, traceback
from typing import List
from index import Index
from util import Util
from chatbot import ChatBot

class Rennat:
    def __init__(self, index_file:str, collection_name:str, openai_token:str) -> None:
        self.index = Index(index_file, collection_name, openai_token)
        self.last_answer = None
        self.last_sources = None
        self.last_papers = None
        self.last_query = None
        self.chatbot = None
        self.n = 25

    def chat(self):
        if not self.chatbot:
            self.chatbot = ChatBot("a research assistant")
        if self.last_query:
            self.chatbot.inform(f"\nto give you some context, this query: {self.last_query}")
        if self.last_answer:
            self.chatbot.inform(f"\n has this tentative answer: {self.last_answer}")
        if self.last_sources:
            self.chatbot.inform(f"\nthese sources were used to generate the answer to the query:")
            remaining_tokens = 2 * (25 + len(self.last_query) + len(self.last_answer))
            i = Util.max_sources(self.last_sources, remaining_tokens)
            for doc in self.last_sources[:i]:
                self.chatbot.inform(f"\n - Source: {doc.metadata['source']}\nContent: {doc.page_content}")
        while True:
            message = input("\nYou (RESET to reset, DONE to exit): ")
            if message == "DONE":
                break
            elif message == "RESET":
                self.chatbot.reset()
            else:
                response = self.chatbot.chat(message)
                # print(f"\n{self.chatbot.name}: {response}") # no need to print when streaming


    def mainmenu(self):
        print("----- User -----")
        print("1. Ask a question")
        print("2. Converse with papers")
        print("3. Chat with a bot")
        print("4. Summarize a paper")
        print("----- Advanced -----")
        print("5. Synthesize literature")
        print("----- Admin -----")
        print("a. List files")
        print("b. List collections")
        print("c. Select/create collection")
        print("d. Add files to collection")
        print("e. Delete files from collection")
        print("f. Delete collection")

        print("DONE Exit")
        choice = input("Please enter your choice: ")
        if choice == '1':
            self.inquire()
        elif choice == '2':
            self.converse()
        elif choice == '3':
            self.chat()
        elif choice == '4':
            self.summarize()
        elif choice == '5':
            self.synthesize()
        elif choice == 'a':
            self.list_files()
        elif choice == 'b':
            self.list_collections()
        elif choice == 'c':
            self.select_collection()
        elif choice == 'd':
            self.add_files()
        elif choice == 'e':
            self.delete_files()
        elif choice == 'f':
            self.delete_collection()
        elif choice == 'DONE':
            sys.exit(0)

    def list_files(self):
        self.current_collection()
        files = self.index.get_file_names()
        print("Files:")
        for f in files:
            print(f)

    def list_collections(self):
        collections = self.index.list_collections()
        print("Collections:")
        for c in collections:
            print(c)

    def select_collection(self):
        collection_name = input("Please enter collection name: ")
        self.index.switch_collection(collection_name)
        self.current_collection("Selected")

    def add_files(self):
        files = input("Please enter files to add (comma separated): ")
        files = files.split(',')
        res = []
        for f in files:
            f = glob(f)
            if f:
                res.extend(f)
        self.index.add_files(res, verbose=True)

    def delete_files(self):
        files = input("Please enter files to delete (comma separated): ")
        files = files.split(',')
        self.index.delete_files(files)

    def delete_collection(self):
        print("WARNING: This will delete the collection and all its contents.")
        print("This action cannot be undone.")
        print("Do you want to continue? (y/n)")
        decision = input()
        if decision == 'y':           
            collection_name = input("Please enter collection name: ")
            self.index.delete_collection(collection_name)

    def number_select(self) -> List[int]:
        "allows the user to input multiple numbers separated by commas, it accepts ranges with dashes as well"
        selected = input()
        if selected:
            selected = selected.split(',')
            #process ranges start-end
            res = []
            for e in selected:
                if '-' in e:
                    start, end = e.split('-')
                    start = int(start)
                    end = int(end)
                    res.extend(range(start, end+1))
                else:
                    res.append(int(e))
            return res
        else:
            return []
    
    def select_from_a_list(self, items:List[str], 
                           prompt1:str = "Based on the meta data, the following documents are found:",
                           prompt2:str = "Select the paper you want to use, enter to select all: "):
        print(prompt1)
        i = 0
        for m in items:
            print(i, '-', m)
            i += 1
        print()        
        print(prompt2)
        selected = self.number_select()
        if selected:
            items = [items[i] for i in selected]
        return items

    def select_papers(self, prompt="Please enter paper keyword, or hit enter to skip: "):
        meta = input(prompt)
        if not meta:
            return
        meta = meta.split(',')
        meta_names = self.index.search_meta(meta)   
        doc_names = self.select_from_a_list(meta_names)
        return doc_names

    def read_query(self, prompt="Please enter your query (DONE to exit): "):
        query = input(prompt)
        if not query:
            return None
        if query.strip() == "DONE":
            return
        modifier = ""
        if query.startswith('%'):
            modifier = query.partition(' ')[0][1:].replace("_", " ")
            query = query.partition(' ')[-1].strip()        
        return query, modifier    
    
    def display_results(self, query, answer, papers, sources):
        self.last_query = query
        self.last_answer = answer
        self.last_sources = sources
        self.last_papers = papers

        print()
        print("\nSELECTED SOURCES:")
        i = 0
        for doc in sources:
            print(i, '-', doc.metadata['source'], doc.page_content)  
            i += 1
            print()

        print("\nSELECTED PAPERS:")
        i = 0
        for paper in papers:
            print(i, '-', paper)
            i += 1

        print("\n\nFINAL ANSWER")
        print(answer)
        print()
    
    def summarize(self):
        meta_names = self.select_papers()
        if not meta_names:
            print("You have to select source papers")
            return
        while True:
            query_modifier = self.read_query("enter a query (e.g., summarize)  (DONE to exit): ")
            if not query_modifier:
                break
            query, _ = query_modifier
            sources = self.index.get_documents(meta_names)
            if len(meta_names) == 1:
                # select pages
                pages = {source.metadata['page'] for source in sources}
                min_page = min(pages)
                max_page = max(pages)
                print(f"The selected paper has {len(pages)} pages, from {min_page} to {max_page}")
                print("You can select a subset or press enter to continue with all")
                selected = self.number_select()
                if selected:
                    sources = [source for source in sources if source.metadata['page'] in selected]
            else:
                print(f"summarizing these papers requires processing {len(sources)} sources")
                print("You can select a subset or press enter to continue with all")
                selected = self.number_select()
                if selected:
                    sources = [sources[i] for i in selected]
            text = self.index.refine(query, sources, False)
            self.display_results(query, text, meta_names, sources)

    def synthesize(self):
        while True:
            query_modifier = self.read_query()
            if not query_modifier:
                break
            query, modifier = query_modifier
            docs = None
            while True:
                answer, papers, sources = self.index.synthesize(query, docs)
                self.display_results(query, answer, papers, sources)
                decision = input("do you want to re-answer the question using a subset of the sources? (y/n) ")
                if decision == 'y':
                    print("You can select a subset or press enter to continue with all")
                    selected = self.number_select()
                    if selected:
                        docs = [sources[i] for i in selected]
                else:
                    break

    def inquire(self):
        while True:
            query_modifier = self.read_query()
            if not query_modifier:
                break
            query, modifier = query_modifier
            docs = None
            while True:
                answer, papers, sources = self.index.answer(query, docs, modifier=modifier)
                self.display_results(query, answer, papers, sources)
                decision = input("do you want to re-answer the question using a subset of the sources? (y/n) ")
                if decision == 'y':
                    print("You can select a subset or press enter to continue with all")
                    selected = self.number_select()
                    if selected:
                        docs = [sources[i] for i in selected]
                else:
                    break

    def converse(self):
        meta_names = self.select_papers()
        while True:
            query_modifier = self.read_query()
            if not query_modifier:
                break
            query, modifier = query_modifier
            docs =  self.index.search_docs(query, meta_names=meta_names)
            # obtain a unique list of documents name from metadata
            doc_names = []
            for doc in docs:
                name = doc.metadata['name']
                if not name in doc_names:
                    doc_names.append(name)
            doc_names = doc_names[:self.n]

            prompt1 = "Based on the query," + "and paper criteria" if meta_names else "" + "the following papers are found:"
            doc_names = self.select_from_a_list(doc_names, prompt1=prompt1)

            docs = [doc for doc in docs if doc.metadata['name'] in doc_names]
            
            while True:
                answer, papers, sources = self.index.answer(query, docs, modifier=modifier)
                self.display_results(query, answer, papers, sources)

                decision = input("do you want to re-answer the question using a subset of the sources? (y/n) ")
                if decision == 'y':
                    print("You can select a subset or press enter to continue with all")
                    selected = self.number_select()
                    if selected:
                        docs = [sources[i] for i in selected]
                else:
                    break

    def current_collection(self, prompt="Current"):
        print(prompt, "collection:", self.index.collection_name, "with", len(self.index.get_file_names()), "files", self.index.size(), "documents")

if __name__ == "__main__":
    # turn off warnings
    # warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    # check that two arguments are provided if not print usage
    if len(sys.argv) != 3:
        print("Usage: python rennat.py <index_file> <openai_token>")
        index_file = input("Enter the index file name: ").strip()
        openai_token = Util.seek_openai_token().strip()
        if not openai_token:
            openai_token = input("Enter your OpenAI API Token: ")
    else:
        index_file = sys.argv[1]
        openai_token = sys.argv[2]

    rennat = Rennat(index_file, "references", openai_token)
    print("Welcome to Rennat!")
    rennat.current_collection()
    while(True):
        try:
            rennat.mainmenu()
            print()
        except Exception as e:  
            print("Error:", e)
            # print stack trace
            traceback.print_exc(file=sys.stdout)



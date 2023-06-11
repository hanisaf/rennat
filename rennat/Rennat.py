#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from typing import List
from Index import Index, Util
from chatbot import ChatBot

class Rennat:
    def __init__(self, index_file:str, collection_name:str, openai_token:str) -> None:
        self.index = Index(index_file, collection_name, openai_token)
        self.last_answer = None
        self.last_sources = None
        self.last_papers = None
        self.last_query = None
        self.chatbot = None

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
            message = input("\nYou (DONE to exit): ")
            if message == "DONE":
                break
            response = self.chatbot.chat(message)
            # print(f"\n{self.chatbot.name}: {response}") # no need to print when streaming


    def mainmenu(self):
        print("1. Ask a question")
        print("2. Chat with a bot")
        print("3. Exit")
        choice = input("Please enter your choice: ")
        if choice == '1':
            self.inquire()
        elif choice == '2':
            self.chat()
        elif choice == '3':
            sys.exit(0)

    def inquire(self, n=25):
        query = input("Please enter your query: ")
        meta = input("Please enter meta data criteria, enter to skip: ")
        if meta:
            meta = meta.split(',')

        modifier = ""
        if query.startswith('%'):
            modifier = query.partition(' ')[0][1:].replace("_", " ")
            query = query.partition(' ')[-1].strip()

        meta_names = None
        if meta:
            meta_names = self.index.search_meta(meta)
            print("Based on the meta data, the following documents are found:")
            i = 0
            for m in meta_names:
                print(i, '-', m)
                i += 1
        print()

        docs =  self.index.search_docs(query, meta_names=meta_names)
        # obtain a unique list of documents name from metadata
        doc_names = []
        for doc in docs:
            name = doc.metadata['name']
            if not name in doc_names:
                doc_names.append(name)
        doc_names = doc_names[:n]
        print("Based on the query,", "and meta criteria" if meta_names else "","the following documents are found:")
        i = 0
        for d in doc_names:
            print(i, '-', d)
            i += 1

        print()
        print("Select the documents you want to use, enter to select all: ")
        selected = input()
        if selected:
            selected = selected.split(',')
            selected = [int(s) for s in selected]
            doc_names = [doc_names[i] for i in selected]
            docs = [doc for doc in docs if doc.metadata['name'] in doc_names]
        
        answer, papers, sources = self.index.answer(query, docs, modifier=modifier)
        self.last_query = query
        self.last_answer = answer
        self.last_sources = sources
        self.last_papers = papers

        print()
        print("\nSELECTED QUOTES:")
        for doc in sources:
            print('-', doc.metadata['source'], doc.page_content)  
            print()

        print("\nSELECTED SOURCES:")
        i = 0
        for paper in papers:
            print(i, '-', paper)
            i += 1

        print("\n\nFINAL ANSWER")
        print(answer)

if __name__ == "__main__":
    rennat = Rennat(os.getenv("RENNAT"), "references", os.getenv("OPENAI_API_KEY"))
    print("Welcome to Rennat!")
    while(True):
        try:
            rennat.mainmenu()
            print()
        except Exception as e:  
            print("Error:", e)


#%% 

import os
from typing import List
from Index import Index
#%%
class Rennat:
    def __init__(self, index_file:str, collection_name:str, openai_token:str) -> None:
        self.index = Index(index_file, collection_name, openai_token)

    def inquire(self, query: str = None, meta: str | List[str] = None, n=25, short=False, interactive=True):
        if not query:
            # run in interactive mode
            interactive = True
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

        if interactive:
            print()
            print("Select the documents you want to use, enter to select all: ")
            selected = input()
            if selected:
                selected = selected.split(',')
                selected = [int(s) for s in selected]
                doc_names = [doc_names[i] for i in selected]
                docs = [doc for doc in docs if doc.metadata['name'] in doc_names]
        
        text, papers, sources = self.index.answer(query, docs, modifier=modifier)


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
        print(text)

        if not interactive:
            return text

# %%

# testing
if __name__ == "__main__":
    rennat = Rennat(os.getenv("RENNAT"), "references", os.getenv("OPENAI_API_KEY"))
    while(True):
        rennat.inquire()
        print()

    #rennat.inquire("what are affordances?", ["karahanna 2018", "leonardi 2015"])
#%%


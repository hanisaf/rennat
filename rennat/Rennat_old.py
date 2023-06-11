#%% 

import os
from typing import List
import utils
import prompts
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

#%%
class Rennat:
    def __init__(self, index_file:str, openai_token:str) -> None:
        utils.load_env(openai_token)
        self.open_ai_token = os.getenv("OPENAI_API_KEY")
        self.index = utils.load_store(index_file)
        self.llm = utils.get_llm()
        pass

    def inquire(self, query: str = None, meta: str | List[str] = None, n=25, short=False, interactive=True):
        if not query:
            # run in interactive mode
            interactive = True
            query = input("Please enter your query: ")
            meta = input("Please enter meta data criteria, enter to skip: ")
            if meta:
                meta = meta.split(',')
        meta_names = None
        if meta:
            meta_names = utils.search_meta(self.index, meta)
            print("Based on the meta data, the following documents are found:")
            i = 0
            for m in meta_names:
                print(i, '-', m)
                i += 1
        print()
        k = len(self.index.docstore._dict)
        docs = utils.search_docs(self.index, query, k=k, meta_names=meta_names)
        # obtain a unique list of documents name from metadata
        doc_names = []
        for doc in docs:
            name = doc.metadata['name']
            if not name in doc_names:
                doc_names.append(name)
        # if meta_names:
        #     doc_names = [name for name in doc_names if name in meta_names]
        doc_names = doc_names[:n]
        print("Based on the query, the following documents are found:")
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

        # calculate the maximum number of words that can be added
        max_words = 4096 / 3
        prompt_words = len(prompts.BASE_PROMPT.split())
        remaining_words = max_words - prompt_words
        i = 0
        for doc in docs:
            doc_words = len(doc.page_content.split())
            if doc_words < remaining_words:
                i += 1
                remaining_words -= doc_words
        sources = docs[:i-1]

        if short:
            prompt = prompts.SHORT_QA_PROMPT
        else:
            prompt = prompts.QA_PROMPT
        chain = load_qa_with_sources_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=False,)
        
        print()

        answer = chain(
            {"input_documents": sources, "question": query}, return_only_outputs=False
        )

        text = answer["output_text"].strip()

        print()
        print("\nSELECTED QUOTES:")
        for doc in sources:
            print('-', doc.metadata['source'], doc.page_content)  
            print()

        print("\nSELECTED SOURCES:")
        papers = []
        i = 0
        for doc in sources:
            if not doc.metadata['name'] in papers:
                papers.append(doc.metadata['name'])
                print(i, '-', doc.metadata['name'])
                i += 1

        print("\n\nFINAL ANSWER")
        print(text)

        if not interactive:
            return text

# %%

# testing
if __name__ == "__main__":
    rennat = Rennat(os.getenv("REFERENCES"), os.getenv("OPENAI_API_KEY"))
    while(True):
        rennat.inquire()
        print()
    # rennat.inquire("what are affordances?", ["karahanna 2018", "leonardi 2015"])
#%%


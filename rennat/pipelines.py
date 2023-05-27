import prompts
from utils import *


def inquire(query, index, sources=None, temperature=0.0, k=5, 
            chain_type="stuff", prompt=prompts.QA_PROMPT, verbose=False,
            print_sources=True):
    llm = get_llm(temperature=temperature)
    if not sources:
        sources = search_docs(index, query, k=k)
    chain = load_qa_with_sources_chain(
        llm,
        chain_type=chain_type,
        prompt=prompt,
        verbose=verbose,)
    
    answer = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=False
    )

    text = answer["output_text"].strip()

    print("\n\nFINAL ANSWER")
    print(text)

    if(print_sources):
        print("\nPROVIDED SOURCES:")
        for doc in sources:
            print(doc.metadata['source'], doc.page_content)  
            print()
    return text


def refine(query, index, temperature=0.0, k=5, verbose=False,
            print_sources=True):
    llm = get_llm(temperature=temperature)
    sources = search_docs(index, query, k=k)
    chain = load_qa_with_sources_chain(llm, chain_type="refine", verbose=verbose)
    answer = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    text = answer["output_text"].strip()

    print("\n\nFINAL ANSWER:")
    print(text)

    if(print_sources):
        print("\nPROVIDED SOURCES:")
        for doc in sources:
            print(doc.metadata['source'], doc.page_content) 
            print()
    return text 


def respond(issue, index, temperature=0.0, k=5, 
            chain_type="stuff", prompt=prompts.REBUT_PROMPT, verbose=False,
            print_sources=True):
    return inquire(issue, index, temperature, k, chain_type, prompt, verbose, print_sources)
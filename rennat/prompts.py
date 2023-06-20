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

    CHAT_REFINE_PROMPT = (
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
#TODO
"Source citation: (Stendal, Thapa, Lanamaki, p. 8):Given actors, have  the capabilities to perform the action. Likewise, we  found another point of confusion in the identification,  for example how and who identify the affordances, likewise when do the affordances identified.   The question of affordances as possibility of  action or affordance as performed action remains  unclear. The literatures reported in this paper have  reported list of affordan ces, but also insisted on  understanding the actualization and consequences of  identified affordances.   Finally, we agree that affordance is a useful lens  to understand the sociotechnical mechanism in  Information System context, though it needs critical  construction to progress towards maturity."


# adapted from https://github.com/mmz-001/knowledge_gpt

from utils import create_template

BASE_PROMPT = """Create a /length/ final answer to the given questions using the provided sources.

QUESTION: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:"""

SHORT_QA_PROMPT =  create_template(BASE_PROMPT.replace("/length/", "short"))


QA_PROMPT =  create_template(BASE_PROMPT.replace("/length/", "detailed"))

DETAILED_PROMPT = create_template("""Create an elaborate final answer in bullet list form to the given questions using the provided sources.

QUESTION: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:

bullet list:
""")

REBUT_PROMPT = create_template("""Create an elaborate final answer to rebut the given issues using the provided sources. 

ISSUES: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:""")

DETAILED_REBUT_PROMPT = create_template("""Create an elaborate answer in bullet list form to rebut the given issues using the provided document sources. 

ISSUES: {question}
=========
SOURCES:

{summaries}
=========
FINAL ANSWER:

bullet list:""")
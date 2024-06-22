from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

############------ I was playing around with a persistant GPT embedding------###########
# from embedWork import persist_dir, embedding

#LLM Setup
# local_llm = "SocBot"
local_llm = "llama3"

# vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
loader = CSVLoader(file_path="../AppraiSetRAG.csv")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
data_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=data_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
)
retriever = vectorstore.as_retriever()

### Generater
prompt2 = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an art history tutor. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Limit your response to 4 sentences. End every answer with one relevant question. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

llm2 = ChatOllama(model=local_llm, temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt2 | llm2 | StrOutputParser()

# Run
docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

def fromRAG_gen(question):
    docs = retriever.invoke(question)
    # doc_txt = docs[1].page_content
    generation = rag_chain.invoke({"context": docs, "question": question})
    return generation

########################----- If we want to create more of a workflow in the system for reviewing rag content. -----################################

# Retrieval Grader
llm = ChatOllama(model=local_llm, format="json", temperature=0)
prompt1 = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = prompt1 | llm | JsonOutputParser()
question = "condition report"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### Hallucination Grader
prompt3 = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt3 | llm | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})


### Answer Grader
prompt4 = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = prompt4 | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})

### Router
prompt5 = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on art, auctions, and art history. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
question_router = prompt5 | llm | JsonOutputParser()
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
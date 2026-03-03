# rag.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from transformers import pipeline


# Role based docs
documents = {
    "finance": ["Finance report: Company revenue is 5M"],
    "marketing": ["Marketing report: Campaign A performed best"],
    "hr": ["HR report: Employee attendance is 95%"],
    "engineering": ["Engineering doc: System uses microservices"],
    "employee": ["Company policy: Work from home allowed twice a week"],
    "executive": ["Executive report: All departments summary"]
}


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstores = {}

# Create vectorstore for each role
for role, docs in documents.items():
    vectorstores[role] = FAISS.from_texts(docs, embedding)


# Load local model
pipe = pipeline("text-generation", model="distilgpt2")

llm = HuggingFacePipeline(pipeline=pipe)


def get_answer(role, question):

    if role not in vectorstores:
        return "Access denied"

    retriever = vectorstores[role].as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa.run(question)

    return result
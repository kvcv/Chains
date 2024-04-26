import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Load all the .pdf files from docs directory
pdf_file = open('Choosing an effective visual.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)
pages=len(pdf_reader.pages)
text = []

# Extract text from each page
for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    ext = page.extract_text()
    text.append(ext)

### Embeddings and vector store -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key="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")

# VectorStore ------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.create_documents(text)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

template = """Answer the question based on the context below. If you can't answer the question, reply "I don't know".
    {context}

    Question: {input}
    """
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()
model = ChatOpenAI(openai_api_key="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")
gen_chain = {"context": retriever, "input": RunnablePassthrough()} | prompt | model | parser

print(gen_chain.invoke("who is the author?"))
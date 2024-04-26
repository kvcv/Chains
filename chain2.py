import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain

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


# Create chain -------------------------------------------
# this template takes "context" and "question" to be taken in as variables
template = """Answer the question based only on the following context:
    {context}

    Question: {input}
    """
prompt = PromptTemplate.from_template(template)
model = OpenAI(openai_api_key="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")

chain = prompt | model 

# Retrieval Chain -------------------------------------------
retriever = vector.as_retriever(search_type="similarity")
#document_chain here is an instance of create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, chain)
response = retrieval_chain.invoke({"input": "how choose a visual?"})
print(response["answer"])
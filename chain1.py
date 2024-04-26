import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
"""
from langchain_community.document_loaders import DirectoryLoader
pdf_loader = DirectoryLoader('/content/Documents/', glob="**/*.pdf")
readme_loader = DirectoryLoader('/content/Documents/', glob="**/*.md")
txt_loader = DirectoryLoader('/content/Documents/', glob="**/*.txt")
     

#take all the loader
loaders = [pdf_loader, readme_loader, txt_loader]

#lets create document 
documents = []
for loader in loaders:
    documents.extend(loader.load())"""
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
# prompts are given to chain classes to instantiate them
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

llm = ChatOpenAI(openai_api_key="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")

# Create chain, all chains need at least an LLM and a prompt
document_chain = create_stuff_documents_chain(llm, prompt)

response = document_chain.invoke({
    "input": "What is Interactivity?",
    "context": [Document(page_content="What")]
})
print(response)
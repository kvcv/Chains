# pip install -U langchain langchain-community langchain-openai
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
import sqlite3

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name = llm_model, temperature = 0, api_key="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")
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "how many employees do you have?"})
print(response)
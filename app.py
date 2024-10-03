# Importing libraries
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Load environment variables from the .env file
load_dotenv()

# Retrieve PostgreSQL database credentials from environment variables
POSTGRESQL_HOST = os.getenv('POSTGRESQL_HOST')
POSTGRESQL_USER = os.getenv('POSTGRESQL_USER')
POSTGRESQL_PASS = os.getenv('POSTGRESQL_PASS')
POSTGRESQL_DB = os.getenv('POSTGRESQL_DB')

# Construct the PostgreSQL URI for SQLAlchemy
POSTGRESQL_URI = f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASS}@{POSTGRESQL_HOST}:5432/{POSTGRESQL_DB}"

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configure OpenAI ChatGPT
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

# Function to configure and connect to the PostgreSQL database
def configure_db(db_uri):
    return SQLDatabase(create_engine(db_uri))

# Create a database connection object using the SQLAlchemy engine
db = configure_db(POSTGRESQL_URI)

# Create a ConversationBufferMemory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a custom prompt template that includes the chat history
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Don't answer any question if you don't have the answer in database, instead say "I don't have the answer to that question".
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQL query"
Answer: "Final answer here"

Only use the following tables:
{table_info}

If someone asks for the value of a field, don't return the entire row but just the value of the field.

Chat History:
{chat_history}

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "chat_history"],
    template=_DEFAULT_TEMPLATE
)

# Create a SQLDatabaseChain object
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    memory=memory,
    prompt=PROMPT,
    verbose=True
)

# Function to handle user queries
def handle_query(query):
    try:
        response = db_chain.invoke(query)
        result = response.get('result', '')
        
        # Extract the actual SQL query from the response
        sql_query_start = result.find('SQLQuery:') + 9
        sql_query_end = result.find('SQLResult:')
        if sql_query_start != -1 and sql_query_end != -1:
            sql_query = result[sql_query_start:sql_query_end].strip()
            
            # Remove any markdown formatting
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Execute the cleaned SQL query
            with db.engine.connect() as connection:
                result_set = connection.execute(sql_query)
                rows = result_set.fetchall()
            
            # Format the result
            formatted_result = "\n".join([str(row) for row in rows])
            print(f"Query: {query}")
            print(f"SQL Query: {sql_query}")
            print(f"Result: {formatted_result}")
        else:
            print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
query1 = "Who has taken the most wickets?"
handle_query(query1)

query2 = "How many overs did he bowl?"
handle_query(query2)
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from macros import *
from modules.Path import loadPath
from modules import Functions

FileSystemManager = Functions.FileSystemManager()
ProcessManager = Functions.ProcessManager()

loadPath(ProcessManager)

llm = Ollama(model="mistral")


response_schemas = [
    ResponseSchema(name="action", description="The most suitable action to achieve this task for the user"),
    ResponseSchema(name="parameters", description="The parameters accompanying the action to achieve the desired goal")
]

code_schemas = [
    ResponseSchema(name="code", description="The code for given program"),
    ResponseSchema(name="language", description="The programming language used to write the code")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
code_parser = StructuredOutputParser.from_response_schemas(code_schemas)

format_instructions = output_parser.get_format_instructions()
code_instructions = code_parser.get_format_instructions()

mainTemplate = """
You will take user query and based on that select most suitable action.
1) SearchWeb or 2) WriteCode 3) Converse
and give output in the following template
{format_instructions}

Query: {query}
"""
codeTemplate = """
You are Luna, a virtual assistant. Write code for the user query. Don't write anything else except the code in mentioned programming language.

Query: {query}
"""
chatTemplate = """
You are Luna, a virtual assistant. You were made by Kevin Santhosh. Your task is to analyze user query and provide step by step instructions to solve their query.
Query: {query}
"""

prompt = PromptTemplate(template=mainTemplate, input_variables=["format_instructions", "query"])
codePrompt = PromptTemplate(template=codeTemplate, input_variables=["query"])
chatPrompt = PromptTemplate(template=chatTemplate, input_variables=["query"])

st.title("LUNA: Virtual Assistant")
user_prompt = st.text_area("Enter your prompt")
button = st.button("Submit")

if button:
    if user_prompt:
        finalPrompt = prompt.format(query=user_prompt, format_instructions=format_instructions)
        output = llm.invoke(finalPrompt)
        data = output_parser.parse(output)

        if data['action'] == 'SearchWeb':
            st.write("Search Web for:", data['parameters'])
            searchWeb(data['parameters'])
        elif data['action'] == 'WriteCode':
            st.write("Sure, I will write the code for you!")
            codeOutput = llm.invoke(codePrompt.format(query=user_prompt))
            st.write("Code API")
            st.write(codeOutput)
            lines = codeOutput.split('\n')
            lines.pop(0)
            lines.pop()
            codeOutput = '\n'.join(lines)
            FileSystemManager.writeToFile("code.txt", codeOutput)
        else:
            st.write("[Chat Mode]")
            st.write(llm.invoke(chatPrompt.format(query=user_prompt)))
        
        st.write("API Data (Currently in development)")
        st.write(output)
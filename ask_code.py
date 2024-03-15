from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings

from code_assistant import CodeAssistant
from code_indexer import CodeIndexer

EMBEDDING_MODEl = "mistral:latest"
CODE_INDEX = "code_index"
CODE_REPOSITORY_FOLDER = 'code_repository'

if __name__ == '__main__':
    load_dotenv()
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEl)
    chat_model = ChatOpenAI(model_name="gpt-4")

    indexer = CodeIndexer(embedding_model)
    indexer.index(CODE_REPOSITORY_FOLDER, CODE_INDEX)

    code_assistant = CodeAssistant(CODE_INDEX, chat_model, embedding_model)

    while True:
        user_input = input("You: ")

        response = code_assistant.ask_question(user_input)
        print("Code Assistant:", response)

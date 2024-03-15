from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import Output


class CodeAssistant:
    def __init__(self, local_index, chat_model, embedding_model):
        index = FAISS.load_local(local_index, embedding_model, allow_dangerous_deserialization=True)
        retriever = index.as_retriever(search_kwargs={"k": 7})

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a code assistant that helps user to understand existing code and write a new code based on provided context.
                    Context: {context}
                    Always provide the name of sources were you found the information, using this format:
                    - source 1
                    - source 2
                    - source N  
                    """,
                ),

                ("human", "{question}"),
            ]
        )

        self.chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | chat_model
                | StrOutputParser()
        )

    def ask_question(self, question) -> Output:
        return self.chain.invoke(question)

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS


class CodeAssistant:
    def __init__(self, local_index, chat_model, embedding_model):
        index = FAISS.load_local(local_index, embedding_model, allow_dangerous_deserialization=True)
        retriever = index.as_retriever(search_kwargs={"k": 7})

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            memory=memory,
            retriever=retriever,
            return_source_documents=True
        )

    def ask_question(self, question) -> str:
        return self.chain.invoke(question).get("answer")

import time

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS


class CodeIndexer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def index(self, folder_path, index_name):
        print(f"Indexing code files from {folder_path}...")
        start_time = time.time()

        loader = DirectoryLoader(folder_path, glob="**/*.java", show_progress=True, loader_cls=TextLoader)

        docs = loader.load()

        print(f"Total number of documents: {len(docs)}")

        for doc in docs:
            print(doc.metadata)

        embedding_index = FAISS.from_documents(docs, self.embedding_model)
        embedding_index.save_local(index_name)

        print(f"Indexing total time taken: {time.time() - start_time} seconds")

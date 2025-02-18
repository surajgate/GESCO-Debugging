import traceback
import os
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.documents import Document
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv()
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "CHATGPT")

OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL")
AZURE_OPENAI_EMBEDDINGS_MODEL = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_MODEL", "")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")


def connect_to_milvus(alias, connection_string):
    """
    Establish connection to Milvus server
    """
    host, port = connection_string.split(':')
    connections.connect(alias=alias, host=host, port=port)


def embed_docs(source_alias, source_collection_name, target_collection_name, llm_type):
    """
    Copy data from source collection, re-embed it with hybrid embeddings 
    and store in target collection
    """
    # Connect to source collection
    source_collection = Collection(
        name=source_collection_name, using=source_alias)
    source_collection.load()
    batch_size = 500
    total_entities = source_collection.num_entities
    process_status_flag = False
    for offset in range(0, total_entities, batch_size):
        output_fields = ["file_directory", "filename", "page_number", "orig_elements",
                         "fileid", "text"]

        # Search with empty expression to get all data
        results = source_collection.query(
            expr="",
            output_fields=output_fields,
            limit=batch_size,
            offset=offset
        )

        if results:
            # Process each entity individually
            doc_list = []
            process_status_flag = False
            for entity in results:
                page_content = entity["text"],
                meta_data = {
                    "file_directory": entity["file_directory"],
                    "filename": entity["filename"],
                    "page_number": entity["page_number"],
                    "orig_elements": entity["orig_elements"],
                    "fileid": entity["fileid"],
                }
                try:
                    doc_list.append(
                        Document(page_content=page_content[0], metadata=meta_data))
                except Exception as e:
                    stacktrace = traceback.format_exc()
                    print(f"page content :\n\n{page_content}\n\n")
                    print(f"metadata :\n\n{meta_data}\n\n")
                    print(f"stacktrace :\n\n{stacktrace}\n\n")
                    raise e

            if llm_type == 'CHATGPT':
                embedding_function = OpenAIEmbeddings(
                    model=OPENAI_EMBEDDINGS_MODEL)
            elif llm_type == 'AZURE-CHATGPT':
                embedding_function = AzureOpenAIEmbeddings(
                    api_key=AZURE_OPENAI_API_KEY,
                    model=AZURE_OPENAI_EMBEDDINGS_MODEL,
                    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
                )
            _ = Milvus.from_documents(
                collection_name=target_collection_name,
                documents=doc_list,
                embedding=embedding_function,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={
                    "uri": MILVUS_CONNECTION_URI,
                },
                consistency_level="Strong",
            )
            print(
                f"Processed {min(offset + batch_size, total_entities)}/{total_entities} entities")
            process_status_flag = True

    return process_status_flag


def main():
    # Connection details
    # Replace with your server connection string
    server_connection = "localhost:19530"
    source_collection_name = input("Enter source collection name:")
    target_collection_name = input("Enter target collection name:")
    llm_type = input("Enter type of llm (eg: CHATGPT/AZURE-CHATGPT):")

    try:
        # Connect to source Milvus instance
        connect_to_milvus("source", server_connection)

        # Copy collection
        process_status_flag = embed_docs(
            "source", source_collection_name, target_collection_name, llm_type)

        if process_status_flag:
            print("Hybrid collection created successfully")
        else:
            print("Failed to create hybrid collection")

    except Exception as e:
        stacktrace = traceback.format_exc()
        print(stacktrace)
        print(f"Error occurred: {str(e)}")

    finally:
        # Clean up connections
        connections.disconnect("source")


if __name__ == "__main__":
    main()

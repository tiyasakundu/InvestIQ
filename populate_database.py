from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from get_embedding_function import get_embedding_function
import numpy as np
import os
import shutil
import json
import camelot as cam

DATA_PATH = os.path.join('.', 'data', 'MF_data')
CHROMA_PATH = 'chroma'

YEARS = [2022, 2023, 2024]

def main():
    create_database()

def create_database():
    '''
    for year in YEARS:
        pdfDataframes = load_pdf_dataframes(year)
        print(pdfDataframes[0].df)
        # big_dict = convert_to_json(documents, year)
        # with open(os.path.join(DATA_PATH, f'MF Data - March {year} - April {year - 1}.json'), 'w') as jj:
        #     json.dump(
        #         big_dict,
        #         jj,
        #         indent=4
        #     )
        # json_file_path = os.path.join(DATA_PATH, f'MF Data - March {year} - April {year - 1}.json')    
        save_to_csv(pdfDataframes, year)
    '''
    
    documents = load_csv_documents()
    save_to_chroma(documents)
    # chunks = split_text(documents)
    # save_to_chroma(chunks)

def load_pdf_dataframes(year: int):
    file_path=(os.path.join(DATA_PATH, f'MF Data - March {year} - April {year - 1}.pdf'))
    dfs = cam.read_pdf(
        file_path,pages='all',flavor='lattice',strip_text='\n',split_text=True)
    for i in range(len(dfs)):
        dfs[i].df.columns = dfs[i].df.iloc[1]
    return dfs

def split_text(documents: list[Document]):
    # loader = JSONLoader(json_file_path, jq_schema=".[] | .[]")
    # documents = loader.load()

    # with open(json_file_path, 'r') as f:
    #     data = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

    chunks = []
    for item in data:
        text = data[item]
        item_chunks = text_splitter.split_text(text)
        chunks.extend(item_chunks)

    print(f"Split {len(data)} items into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def convert_to_json(dfs, year):
    date_range = [
        f'March {year}',
        f'February {year}',
        f'January {year}',
        f'December {year-1}',
        f'November {year-1}',
        f'October {year-1}',
        f'September {year-1}',
        f'August {year-1}',
        f'July {year-1}',
        f'June {year-1}',
        f'May {year-1}',
        f'April {year-1}'
    ]
    big_dict = {}
    for i in range(len(date_range)):
        big_dict[date_range[i]] = dfs[i].df.replace({np.nan: None}).to_dict(orient='records')
    return big_dict

def save_to_csv(dfs, year):
    date_range = [
        f'March {year}',
        f'February {year}',
        f'January {year}',
        f'December {year-1}',
        f'November {year-1}',
        f'October {year-1}',
        f'September {year-1}',
        f'August {year-1}',
        f'July {year-1}',
        f'June {year-1}',
        f'May {year-1}',
        f'April {year-1}'
    ]
    for i in range(len(date_range)):
        with open(os.path.join(DATA_PATH, 'csv', f'MF Data - {date_range[i]}.csv'), 'w') as ff:
            dfs[i].df.replace({np.nan: None}).to_csv(ff, index=False)

def load_csv_documents():
    documents = []
    for year in YEARS:
        date_range = [
            f'March {year}',
            f'February {year}',
            f'January {year}',
            f'December {year-1}',
            f'November {year-1}',
            f'October {year-1}',
            f'September {year-1}',
            f'August {year-1}',
            f'July {year-1}',
            f'June {year-1}',
            f'May {year-1}',
            f'April {year-1}'
        ]
        for date in date_range:
            loader = CSVLoader(
                os.path.join(DATA_PATH, 'csv', f'MF Data - {date}.csv')
                )
            documents.extend(loader.load())
    return documents

if __name__ == '__main__':
    main()
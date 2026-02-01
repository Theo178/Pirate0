import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile

class ScriptProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    async def process_file(self, file: UploadFile) -> str:
        content = ""
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            # Save temporarily to read (or read stream directly if pypdf supports bytes)
            # pypdf can read from bytes stream
            pdf = PdfReader(file.file)
            for page in pdf.pages:
                content += page.extract_text() + "\n"
        elif filename.endswith('.txt') or filename.endswith('.md'):
            content = (await file.read()).decode('utf-8')
        else:
            raise ValueError("Unsupported file type")
            
        return content

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

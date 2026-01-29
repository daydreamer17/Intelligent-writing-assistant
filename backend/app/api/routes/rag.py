from __future__ import annotations

from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...models.schemas import (
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    SourceDocumentResponse,
    UploadDocumentsRequest,
    UploadDocumentsResponse,
)
from ...services.rag_service import UploadedDocument
from ..deps import AppServices, get_services

router = APIRouter(tags=["rag"])


@router.post("/rag/upload", response_model=UploadDocumentsResponse)
def upload_documents(
    payload: UploadDocumentsRequest,
    services: AppServices = Depends(get_services),
) -> UploadDocumentsResponse:
    try:
        max_bytes = services.upload_max_bytes
        docs = [
            UploadedDocument(
                doc_id=item.doc_id,
                title=item.title,
                content=item.content,
                url=item.url,
            )
            for item in payload.documents
        ]
        for doc in docs:
            if len(doc.content.encode("utf-8", errors="ignore")) > max_bytes:
                raise HTTPException(status_code=413, detail="Document too large")
        added = services.rag.add_documents(docs)
        response_docs = [
            SourceDocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                url=doc.url,
            )
            for doc in added
        ]
        return UploadDocumentsResponse(documents=response_docs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Upload failed") from exc


@router.post("/rag/search", response_model=SearchDocumentsResponse)
def search_documents(
    payload: SearchDocumentsRequest,
    services: AppServices = Depends(get_services),
) -> SearchDocumentsResponse:
    try:
        results = services.rag.search(payload.query, top_k=payload.top_k)
        response_docs = [
            SourceDocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                url=doc.url,
            )
            for doc in results
        ]
        return SearchDocumentsResponse(documents=response_docs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Search failed") from exc


@router.post("/rag/upload-file", response_model=UploadDocumentsResponse)
async def upload_files(
    files: list[UploadFile] = File(..., description="上传文本文件"),
    services: AppServices = Depends(get_services),
) -> UploadDocumentsResponse:
    try:
        max_bytes = services.upload_max_bytes
        documents: list[UploadedDocument] = []
        for file in files:
            content_bytes = await file.read()
            if len(content_bytes) > max_bytes:
                raise HTTPException(status_code=413, detail="File too large")
            content = _extract_text(file.filename or "", content_bytes)
            documents.append(
                UploadedDocument(
                    doc_id=str(uuid4()),
                    title=file.filename or "uploaded",
                    content=content,
                    url="",
                )
            )
        added = services.rag.add_documents(documents)
        response_docs = [
            SourceDocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                url=doc.url,
            )
            for doc in added
        ]
        return UploadDocumentsResponse(documents=response_docs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="File upload failed") from exc


def _extract_text(filename: str, content_bytes: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf_text(content_bytes)
    if name.endswith(".docx"):
        return _extract_docx_text(content_bytes)
    return content_bytes.decode("utf-8", errors="ignore")


def _extract_pdf_text(content_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise HTTPException(status_code=400, detail="PDF support not installed") from exc
    reader = PdfReader(BytesIO(content_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_docx_text(content_bytes: bytes) -> str:
    try:
        import docx
    except Exception as exc:
        raise HTTPException(status_code=400, detail="DOCX support not installed") from exc
    file_like = BytesIO(content_bytes)
    document = docx.Document(file_like)
    parts = [para.text for para in document.paragraphs if para.text]
    return "\n".join(parts)

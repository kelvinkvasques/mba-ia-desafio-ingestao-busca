"""Uso: python src/ingest.py."""
import hashlib
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppError, Settings
from errors import error_message
from storage import Repository


def load_chunks(path):
    if not path.is_file():
        raise AppError('PDF ausente. Coloque document.pdf na raiz ou configure PDF_PATH no .env.')
    try:
        data = path.read_bytes()
        pages = PyPDFLoader(str(path)).load()
        # Detecta substituição do arquivo durante a leitura.
        if path.read_bytes() != data:
            raise AppError('PDF mudou durante a leitura. Execute a ingestão novamente.')
    except AppError:
        raise
    except Exception:
        raise AppError('Não foi possível ler o PDF. Verifique integridade, permissões e criptografia.') from None
    readable = [page for page in pages if page.page_content.strip()]
    if not readable:
        raise AppError('PDF sem texto extraível. Faça OCR antes da ingestão.')
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(readable)
    return len(pages), chunks, hashlib.sha256(data).hexdigest()


def ingest_document(repo, pages, chunks, pdf_hash):
    # Versão inclui também o algoritmo e o modelo: vetores semanticamente diferentes nunca se misturam.
    revision = hashlib.sha256((pdf_hash + '\0' + repo.settings.embedding_model + '\0split-1000-150-v1').encode()).hexdigest()
    with repo.ingestion_lock():
        active = repo.active()
        if active and active['revision'] == revision:
            return False
        for index, chunk in enumerate(chunks):
            chunk.metadata['revision'] = revision
            chunk.metadata['embedding_model'] = repo.settings.embedding_model
            chunk.metadata['chunk_index'] = index
        ids = [hashlib.sha256((repo.settings.collection + '\0' + revision + '\0' + str(i)).encode()).hexdigest()
               for i in range(len(chunks))]
        # IDs determinísticos usam UPSERT do PGVector: tentativas após falha não duplicam chunks.
        repo.store.add_documents(chunks, ids=ids)
        # Só publica após gravação completa. Versões anteriores são mantidas, mas não recuperadas.
        repo.activate(revision, pages, len(chunks))
    return True


def ingest_pdf():
    repo = None
    try:
        settings = Settings.load()
        pages, chunks, digest = load_chunks(settings.pdf_path)
        repo = Repository(settings)
        changed = ingest_document(repo, pages, chunks, digest)
        print(f'Páginas: {pages} | Chunks: {len(chunks)} | ' + ('Ingestão concluída.' if changed else 'Documento já ativo; nenhuma duplicação ou chamada de embeddings.'))
        return 0
    except KeyboardInterrupt:
        print('\nIngestão interrompida. A versão ativa anterior foi preservada.')
        return 130
    except Exception as exc:
        print('ERRO: ' + error_message(exc), file=sys.stderr)
        return 1
    finally:
        if repo:
            repo.close()


if __name__ == '__main__':
    sys.exit(ingest_pdf())

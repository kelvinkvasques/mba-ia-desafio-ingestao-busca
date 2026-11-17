"""PGVector e ponteiro transacional para a versão completa do documento."""
from contextlib import contextmanager

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from config import AppError

DDL = '''CREATE TABLE IF NOT EXISTS rag_active_document (
    collection TEXT PRIMARY KEY, revision TEXT NOT NULL,
    embedding_model TEXT NOT NULL, pages INTEGER NOT NULL, chunks INTEGER NOT NULL
)'''


class Repository:
    def __init__(self, settings):
        self.settings = settings
        self.engine = create_engine(settings.database_url, connect_args={'connect_timeout': 5}, hide_parameters=True)
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model,
                                          api_key=settings.api_key, max_retries=2, request_timeout=60)
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = PGVector(embeddings=self.embeddings, connection=self.engine,
                                   collection_name=self.settings.collection, use_jsonb=True,
                                   create_extension=True, pre_delete_collection=False)
        return self._store

    @contextmanager
    def ingestion_lock(self):
        # Lock por coleção durante toda a ingestão; conexão dedicada, sem pool residual.
        with self.engine.connect() as conn:
            conn.execute(text('SELECT pg_advisory_lock(hashtextextended(:name, 0))'),
                         {'name': self.settings.collection})
            conn.commit()
            try:
                with self.engine.begin() as tx:
                    tx.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                    tx.execute(text(DDL))
                yield
            finally:
                conn.execute(text('SELECT pg_advisory_unlock(hashtextextended(:name, 0))'),
                             {'name': self.settings.collection})
                conn.commit()

    def active(self):
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text('SELECT * FROM rag_active_document WHERE collection=:name'),
                                   {'name': self.settings.collection}).mappings().first()
                return dict(row) if row else None
        except ProgrammingError as exc:
            if getattr(exc.orig, 'sqlstate', None) == '42P01':
                return None
            raise

    def activate(self, revision, pages, chunks):
        with self.engine.begin() as conn:
            conn.execute(text('''INSERT INTO rag_active_document
                (collection, revision, embedding_model, pages, chunks)
                VALUES (:collection, :revision, :model, :pages, :chunks)
                ON CONFLICT (collection) DO UPDATE SET revision=EXCLUDED.revision,
                embedding_model=EXCLUDED.embedding_model, pages=EXCLUDED.pages, chunks=EXCLUDED.chunks'''),
                dict(collection=self.settings.collection, revision=revision,
                     model=self.settings.embedding_model, pages=pages, chunks=chunks))

    def require_active(self):
        active = self.active()
        if not active:
            raise AppError('Ingestão ainda não executada nesta coleção. Execute python src/ingest.py.')
        if active['embedding_model'] != self.settings.embedding_model:
            raise AppError('Modelo de embeddings mudou. Execute python src/ingest.py antes de consultar.')
        return active

    def close(self):
        self.engine.dispose()

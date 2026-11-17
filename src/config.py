"""Configuração independente do diretório de execução."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

ROOT = Path(__file__).resolve().parents[1]


class AppError(Exception):
    """Erro com mensagem segura para o terminal."""


@dataclass(frozen=True)
class Settings:
    database_url: str
    api_key: str
    embedding_model: str
    chat_model: str
    collection: str
    pdf_path: Path
    temperature: object = 0.0

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env', override=False)
        names = ('DATABASE_URL', 'OPENAI_API_KEY', 'OPENAI_EMBEDDING_MODEL',
                 'OPENAI_CHAT_MODEL', 'PG_VECTOR_COLLECTION_NAME')
        values = {name: os.getenv(name, '').strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise AppError('Configurações obrigatórias ausentes: ' + ', '.join(missing) + '. Preencha o .env na raiz.')
        try:
            url = make_url(values['DATABASE_URL'])
            if url.drivername != 'postgresql+psycopg' or not url.host or not url.database:
                raise ValueError()
            raw = os.getenv('OPENAI_CHAT_TEMPERATURE', '0').strip()
            temperature = float(raw) if raw else None
            if temperature is not None and not 0 <= temperature <= 2:
                raise ValueError()
        except (ValueError, TypeError, ArgumentError):
            raise AppError('Use DATABASE_URL postgresql+psycopg://usuario:senha@host:porta/banco e temperatura entre 0 e 2 (ou vazia).') from None
        path = Path(os.getenv('PDF_PATH', 'document.pdf'))
        return cls(values['DATABASE_URL'], values['OPENAI_API_KEY'],
                   values['OPENAI_EMBEDDING_MODEL'], values['OPENAI_CHAT_MODEL'],
                   values['PG_VECTOR_COLLECTION_NAME'], (ROOT / path).resolve(), temperature)

"""Nunca imprime exceções do provedor ou URLs com credenciais."""
from openai import AuthenticationError, PermissionDeniedError, RateLimitError, APIConnectionError, APIError, BadRequestError, NotFoundError
from sqlalchemy.exc import SQLAlchemyError
from config import AppError


def error_message(exc):
    if isinstance(exc, AppError):
        return str(exc)
    if isinstance(exc, AuthenticationError):
        return 'Falha de autenticação na OpenAI. Confira OPENAI_API_KEY.'
    if isinstance(exc, PermissionDeniedError):
        return 'Acesso negado pela OpenAI. Confira as permissões do projeto e do modelo.'
    if isinstance(exc, RateLimitError):
        return 'Limite ou cota da API OpenAI atingido. Confira saldo e limites antes de tentar novamente.'
    if isinstance(exc, APIConnectionError):
        return 'Falha de conexão com a OpenAI. Confira sua rede e tente novamente.'
    if isinstance(exc, BadRequestError):
        if exc.param == 'temperature' and exc.code in {'unsupported_value', 'unsupported_parameter'}:
            return ('O modelo de chat rejeitou a temperatura configurada. '
                    'Deixe OPENAI_CHAT_TEMPERATURE= vazio no .env e reinicie o chat. '
                    'Confira também se essa variável está exportada no shell.')
        return 'Requisição rejeitada pela OpenAI. Confira os parâmetros aceitos pelo modelo configurado.'
    if isinstance(exc, NotFoundError):
        return 'Recurso ou modelo não encontrado na OpenAI. Confira os modelos configurados e o acesso do projeto.'
    if isinstance(exc, APIError):
        return 'Falha na API OpenAI. Confira modelos, parâmetros e disponibilidade do serviço.'
    if isinstance(exc, SQLAlchemyError):
        return 'Falha no PostgreSQL. Confira conexão, autenticação, healthcheck e permissão para criar a extensão vector.'
    return 'Falha operacional inesperada. Confira dependências e configuração; nenhum detalhe sensível foi exibido.'

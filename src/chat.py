"""Chat de uma pergunta por vez, sem histórico externo ao contexto recuperado."""
import sys
from config import Settings
from errors import error_message
from search import create_service
from storage import Repository


def main():
    repo = None
    try:
        settings = Settings.load()
        repo = Repository(settings)
        service = create_service(settings, repo)
        print('Pergunte sobre o PDF. Para encerrar: sair, exit ou quit.')
        while True:
            try:
                question = input('PERGUNTA: ').strip()
                if question.lower() in {'sair', 'exit', 'quit'}:
                    break
                if not question:
                    continue
                try:
                    print('RESPOSTA: ' + service.answer(question))
                except Exception as exc:
                    print('ERRO: ' + error_message(exc), file=sys.stderr)
            except (KeyboardInterrupt, EOFError):
                print()
                break
        print('Chat encerrado.')
        return 0
    except KeyboardInterrupt:
        print('\nChat encerrado.')
        return 130
    except Exception as exc:
        print('ERRO: ' + error_message(exc), file=sys.stderr)
        return 1
    finally:
        if repo:
            repo.close()


if __name__ == '__main__':
    sys.exit(main())

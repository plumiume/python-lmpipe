from sphinx.config import Config

class TypehintsFormatter:
    def __call__(self, ann: str, config: Config) -> str:
        return f'``{ann}``'

typehints_formatter = TypehintsFormatter()

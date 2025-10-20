from typing import Literal

FormatLiteral = Literal['.npy', '.csv', '.json'] | None
Formats: tuple[FormatLiteral, ...] = ('.npy', '.csv', '.json', None)

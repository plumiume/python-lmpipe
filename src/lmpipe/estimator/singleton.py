from typing import Any, TypeGuard, Self

class Singleton:
    _value: Self | None = None
    def __new__(cls):
        if cls._value is None:
            cls._value = super().__new__(cls)
        return cls._value
    @classmethod
    def is_self(cls, other: Any) -> TypeGuard[Self]:
        return other is cls._value

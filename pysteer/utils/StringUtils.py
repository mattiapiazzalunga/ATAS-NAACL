import re


class StringUtils:
    @staticmethod
    def trim_string(s):
        return s.strip()

    @staticmethod
    def remove_trailing_period(s):
        return s[:-1] if s.endswith('.') else s

    @staticmethod
    def remove_spaces(s: str) -> str:
        s = StringUtils.trim_string(s)
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    @staticmethod
    def to_lowercase(s):
        return s.lower()
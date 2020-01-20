import re

__all__ = ('snake_case', 'safe_decode',)

def snake_case(input: str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', input)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_').replace('__', '_')


def safe_decode(input: bytes, prefer: str = None) -> str:
    codecs = ['utf-8', 'latin-1', 'ascii']

    if prefer:
        codecs = [prefer] + codecs

    for codec in codecs:
        try:
            return input.decode(codec)
        except UnicodeDecodeError:
            pass

    input.decode('utf-8') # COULD NOT DETERMINE CODEC, RAISE

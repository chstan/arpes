import re

__all__ = ('snake_case',)

def snake_case(input: str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', input)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_').replace('__', '_')
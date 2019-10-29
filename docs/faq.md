# Frequently Asked Questions

## Igor Installation

### Using the suggested invokation I get a pip error

Pip on Windows appears not to like certain archival formats.
While

```pip
 pip install https://github.com/chstan/igorpy.git#egg=igor-0.3.1
```

should work on most systems, you can also clone the repository:

```git
git clone https://github.com/chstan/igorpy
```

And then install into your environment from inside that folder.

```pip
(my arpes env) > echo "From inside igorpy folder"
(my arpes env) > pip install -e .
```


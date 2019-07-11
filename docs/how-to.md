# PyARPES Crash Course and Usage Documentation

The PyARPES documentation is split into two separate parts. The [first section](/how-to) 
(the one you are reading) contains practical usage information for certain types of data
physics analyses, principally as they relate to ARPES. The [second section](/arpes) 
consists of documentation generated from the analysis code source files, and represents a complete&mdash;if 
not always completely documented&mdash;representation of what is available. 

## First Section: Practical Usage Documentation

This is intended as a crash course in how to use PyARPES. It takes you through the first steps of 
[how to configure](/getting-started) a working environment, [loading data](/loading-data), and 
subsequently through some rudimentary analyses that nonethless represent substantial first steps
towards analyzing typical ARPES data.

If you think the documentation would be better served by more or different information, consider 
[contributing](/contributing) the documentation yourself, or just let Conrad know.

## Second Section: Generated Docs, Type Annotations

While not every function and class is fully documented, type annotations and parameter 
information are nonetheless extremely useful when you forget how to invoke a particular function.

If you work in IPython, recall also you can get the function documentation with 

```python
help(some_analysis_function)
```

or

```python
some_analysis_function?
```

although the former will work in more circumstances.
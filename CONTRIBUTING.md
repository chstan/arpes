# Why

Please **do** add useful new analysis routines here. Having a central source of analysis code across the group will make it easier to work across different environments, and allow us all to access cool new analysis techniques.

# How to Contribute

1. You will need a git client, if you don't want to use a terminal, have a look at Github's [GUI Client](https://desktop.github.com/)
2. Write your new analysis code
3. Put it someplace reasonable in line with the project's organizational principles
4. Add convenience accessors on `.T`, `.S`, or `.F` as relelvant
5. If necessary, add imports to `01-common-imports.ipy`
6. Make sure the new code is adequately documented with a [docstring](https://en.wikipedia.org/wiki/Docstring#Python).
7. If you added new requirements, make sure they get added to `requirements.txt`
8. Ensure you have the latest code by `git pull`ing as necessary 
9. `git commit` your change and `git push`


Note also there is a `patterns` submodule inside `arpes`, you can put simple examples of how to use your analysis code in here if you like.

# Notes on Authorship

Some sense of authorship of various parts of the analysis code are automatically retained. `git` remembers every change in the project and who contributed them. You can view this at the terminal with `git blame`, or from the web or Desktop clients.

If you use (or contribute) a large amount of new work that ends up being a substantial part of someone elses' analysis effort, please consider citing (requesting citation) or including the author (requesting authorship) of the analysis code you used as a coauthor, as you would in other situations.

If you would like or feel that it would be beneficial as far as users seeking help, you can also put authorship information into the docstring, but note that the docstring should **not** become a record of changes to a piece of analysis code.

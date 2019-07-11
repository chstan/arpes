# Customization

This section contains some notes about customizing and making it easier 
to work inside PyARPES. A large amount of what is described here will be well 
known to anyone who has worked with Jupyter for a while, but there are also 
some specifics related to PyARPES.

### `local_config.py`

The local configuration allows you to override the settings that are
committed to the repository and therefore shared. You can use this to as
adjust settings on various interactive tools. Have a look at `arpes.config`
for what settings are read by default

    SETTINGS = {
        # contents here
    }

If you want to override defaults, place a copy of a `local_config.py` file
in the repository root (i.e. above the `arpes` directory), or call
`arpes.config.override_settings({...})` with your changes.

### IPython Kernel Customization

If you don't want to have to import everything all the time and you are adverse
to using `arpes.setup` you should customize your IPython session so that it runs 
imports when you first spin up a kernel. There are good directions for how to do 
this online, but a short version is:

1.  Create an IPython profile, use this to start your notebooks
2.  In `~/.ipython/profile_default/` make a folder `startup`
3.  Add the files
    `~/.ipython/profile_default/startup/00-add-arpes-path.py` and
    `~/.ipython/{Your profile}/startup/01-common-imports.ipy` according
    to the templates in `ipython_templates`. See in particular note
    above about setting the environment variable using this file.
4.  Customize

It is important that the filenames you put are such that
`00-add-arpes-path` is lexographically first, as this ensures that it is
executed first. The `.ipy` extension on `01-common-imports.ipy` is also
essential. Ask Conrad if any of this is confusing.


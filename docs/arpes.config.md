# arpes.config module

Store experiment level configuration here, this module also provides
functions for loading configuration in via external files, to allow
better modularity between different projects.

**class arpes.config.WorkspaceManager(workspace=None)**

> Bases: `object`

**arpes.config.attempt\_determine\_workspace(value=None,
permissive=False, lazy=False, current\_path=None)**

**arpes.config.generate\_cache\_files()**

**arpes.config.load\_json\_configuration(filename)**

> Flat updates the configuration. Beware that this doesn’t update nested
> data. I will adjust if it turns out that there is a use case for
> nested configuration

**arpes.config.load\_plugins()**

**arpes.config.override\_settings(new\_settings)**

**arpes.config.update\_configuration(user\_path=None)**

**arpes.config.use\_tex(rc\_text\_should\_use=False)**

**arpes.config.workspace\_matches(path)**

**arpes.config.workspace\_name\_is\_valid(workspace\_name)**

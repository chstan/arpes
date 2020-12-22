arpes.config module
===================

Store experiment level configuration here, this module also provides
functions for loading configuration in via external files, to allow
better modularity between different projects.

**class arpes.config.WorkspaceManager(workspace: Optional\[Any\] =
None)**

> Bases: `object`

**arpes.config.attempt\_determine\_workspace(value=None,
current\_path=None)**

**arpes.config.generate\_cache\_files() -&gt; None**

**arpes.config.load\_json\_configuration(filename)**

> Flat updates the configuration. Beware that this doesn’t update nested
> data. I will adjust if it turns out that there is a use case for
> nested configuration

**arpes.config.load\_plugins() -&gt; None**

**arpes.config.override\_settings(new\_settings)**

**arpes.config.setup\_logging()**

**arpes.config.update\_configuration(user\_path: Optional\[str\] = None)
-&gt; None**

**arpes.config.use\_tex(rc\_text\_should\_use=False)**

**arpes.config.workspace\_matches(path)**

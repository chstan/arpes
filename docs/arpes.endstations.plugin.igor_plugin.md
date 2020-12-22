arpes.endstations.plugin.igor\_plugin module
============================================

**class arpes.endstations.plugin.igor\_plugin.IgorEndstation**

> Bases:
> [arpes.endstations.SingleFileEndstation](arpes.endstations#arpes.endstations.SingleFileEndstation)
>
> A generic file loader for PXT files. This makes no assumptions about
> whether data is from a hemisphere or otherwise, so it might not be
> perfect for all Igor users, but it is a place to start.
>
> `ALIASES = ['IGOR', 'pxt', 'pxp', 'Wave', 'wave']`
>
> `ATTR_TRANSFORMS = {}`
>
> `MERGE_ATTRS = {}`
>
> `PRINCIPAL_NAME = 'Igor'`
>
> `RENAME_KEYS = {}`
>
> **load\_single\_frame(frame\_path: Optional\[str\] = None, scan\_desc:
> Optional\[dict\] = None,**kwargs)\*\*

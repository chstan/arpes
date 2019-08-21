# arpes.utilities.collections module

**arpes.utilities.collections.deep\_equals(a: Any, b: Any) -\> bool**

**arpes.utilities.collections.deep\_update(destination: Any, source:
Any) -\> Dict\[str, Any\]**

> Doesn’t clobber keys further down trees like doing a shallow update
> would. Instead recurse down from the root and update as appropriate.
> :param destination: :param source: :return:

**class arpes.utilities.collections.MappableDict**

> Bases: `dict`
> 
> Like dict except that +, -, \>\>\*\<\<, / are cascaded to values.

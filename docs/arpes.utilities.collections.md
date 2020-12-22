arpes.utilities.collections module
==================================

**class arpes.utilities.collections.MappableDict**

> Bases: `dict`
>
> Like dict except that +, -, &gt;&gt;\*&lt;&lt;, / are cascaded to
> values.

**arpes.utilities.collections.deep\_equals(a: Any, b: Any) -&gt; bool**

**arpes.utilities.collections.deep\_update(destination: Any, source:
Any) -&gt; Dict\[str, Any\]**

> Doesn’t clobber keys further down trees like doing a shallow update
> would. Instead recurse down from the root and update as appropriate.
> :param destination: :param source: :return:

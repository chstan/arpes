# arpes.load\_pxt module

Implements Igor \<-\> xarray interop, notably loading Igor waves and
packed experiment files.

**arpes.load\_pxt.read\_single\_pxt(reference\_path:
Union\[pathlib.Path, str\], byte\_order=None)**

> Uses igor.igorpy to load a single .PXT or .PXP file :return:

**arpes.load\_pxt.read\_separated\_pxt(reference\_path: pathlib.Path,
separator=None, byte\_order=None)**

**arpes.load\_pxt.read\_experiment(reference\_path: Union\[pathlib.Path,
str\],**kwargs)\*\*

> Reads a whole experiment and translates all contained waves into
> xr.Dataset instances as appropriate
> 
>   - Parameters  
>     **reference\_path** –
> 
>   - Returns

**arpes.load\_pxt.find\_ses\_files\_associated(reference\_path:
pathlib.Path, separator: str = 'S')**

> SES Software creates a series of PXT files they are all sequenced with
> \_S\[0-9\]\[0-9\]\[0-9\].pxt *find\_ses\_files\_associated* will
> collect all the files in the sequence pointed to by *reference\_path*
> :param reference\_path: :return:

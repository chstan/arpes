# arpes.utilities.jupyter module

**arpes.utilities.jupyter.get\_full\_notebook\_information() -\>
Optional\[dict\]**

> Javascriptless method to get information about the current Jupyter
> sessions and the one matching this kernel. :return:

**arpes.utilities.jupyter.get\_notebook\_name() -\> Optional\[str\]**

> Gets the unqualified name of the running Jupyter notebook, if there is
> a Jupyter session not protected by password.
> 
> As an example, if you were running a notebook called
> “Doping-Analysis.ipynb” this would return “Doping-Analysis”.
> 
> If no notebook is running for this kernel or the Jupyter session is
> password protected, we can only return None. :return:

**arpes.utilities.jupyter.generate\_logfile\_path() -\> pathlib.Path**

> Generates a time and date qualified path for the notebook log file.
> :return:

**arpes.utilities.jupyter.get\_recent\_logs(n\_bytes=1000) -\>
List\[str\]**

**arpes.utilities.jupyter.get\_recent\_history(n\_items=10) -\>
List\[str\]**

**arpes.utilities.jupyter.wrap\_tqdm(x, interactive=True,
\*args,**kwargs)\*\*

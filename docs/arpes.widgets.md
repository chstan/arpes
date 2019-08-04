# arpes.widgets module

Provides interactive tools based on matplotlib Qt interactive elements.
This are generally primitive one offs that are useful for accomplishing
something quick. As examples:

1.    - *pca\_explorer* lets you interactively examine a PCA
        decomposition  
        or other decomposition supported by
        *arpes.analysis.decomposition*

2.    - *pick\_points*, *pick\_rectangles* allows selecting many
        individual
        
          - points  
            or regions from a piece of data, useful to isolate locations
            to do further analysis.

3.    - *kspace\_tool* allows interactively setting coordinate offset
        for  
        angle-to-momentum conversion.

4.  *fit\_initializer* allows for seeding an XPS curve fit.

All of these return a “context” object which can be used to get
information from the current session (i.e. the selected points or
regions, or modified data). If you forget to save this context, you can
recover it as the most recent context is saved at *arpes.config.CONFIG*
under the key “CURRENT\_CONTEXT”.

There are also primitives for building interactive tools in matplotlib.
Such as DataArrayView, which provides an interactive and updatable plot
view from an xarray.DataArray instance.

In the future, it would be nice to get higher quality interactive tools,
as we start to run into the limits of these ones. But between this and
*qt\_tool* we are doing fine for now.

**arpes.widgets.pick\_rectangles(data,**kwargs)\*\*

**arpes.widgets.pick\_points(data,**kwargs)\*\*

**arpes.widgets.pca\_explorer(pca, data, component\_dim='components',
initial\_values=None, transpose\_mask=False,**kwargs)\*\*

**arpes.widgets.kspace\_tool(data,**kwargs)\*\*

**arpes.widgets.fit\_initializer(data, peak\_type=\<class
'arpes.fits.fit\_models.LorentzianModel'\>,**kwargs)\*\*

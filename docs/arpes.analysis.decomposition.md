# arpes.analysis.decomposition module

**arpes.analysis.decomposition.decomposition\_along(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
axes: List\[str\], decomposition\_cls, correlation=False,**kwargs)\*\*

> Performs a change of basis of your data according to *sklearn*
> decomposition classes. This allows for robust and simple PCA, ICA,
> factor analysis, and other decompositions of your data even when it is
> very high dimensional.
> 
> Generally speaking, PCA and similar techniques work when data is 2D,
> i.e. a sequence of 1D observations. We can make the same techniques
> work by unravelling a ND dataset into 1D (i.e. np.ndarray.ravel()) and
> unravelling a KD set of observations into a 1D set of observations.
> This is basically grouping axes. As an example, if you had a 4D
> dataset which consisted of 2D-scanning valence band ARPES, then the
> dimensions on our dataset would be “\[x,y,eV,phi\]”. We can group
> these into \[spatial=(x, y), spectral=(eV, phi)\] and perform PCA or
> another analysis of the spectral features over different spatial
> observations.
> 
> If our data was called *f*, this can be accomplished with:
> 
> `` ` transformed, decomp =
> decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'],
> PCA) transformed.dims # -> [X, Y, components]``\`
> 
> The results of *decomposition\_along* can be explored with
> *arpes.widgets.pca\_explorer*, regardless of the decomposition class.
> 
>   - Parameters
>     
>       -   - **data** – Input data, can be N-dimensional but should
>             only  
>             include one “spectral” axis.
>     
>       -   - **axes** – Several axes to be treated as a single axis  
>             labeling the list of observations.
>     
>       -   - **decomposition\_cls** – A sklearn.decomposition class
>             (such  
>             as PCA or ICA) to be used to perform the decomposition.
>     
>       -   - **correlation** – Controls whether StandardScaler() is
>             used  
>             as the first stage of the data ingestion pipeline for
>             sklearn.
>     
>       - **kwargs** –
> 
>   - Returns

**arpes.analysis.decomposition.pca\_along(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
axes: List\[str\], decomposition\_cls, correlation=False,**kwargs)\*\*

> Performs a change of basis of your data according to *sklearn*
> decomposition classes. This allows for robust and simple PCA, ICA,
> factor analysis, and other decompositions of your data even when it is
> very high dimensional.
> 
> Generally speaking, PCA and similar techniques work when data is 2D,
> i.e. a sequence of 1D observations. We can make the same techniques
> work by unravelling a ND dataset into 1D (i.e. np.ndarray.ravel()) and
> unravelling a KD set of observations into a 1D set of observations.
> This is basically grouping axes. As an example, if you had a 4D
> dataset which consisted of 2D-scanning valence band ARPES, then the
> dimensions on our dataset would be “\[x,y,eV,phi\]”. We can group
> these into \[spatial=(x, y), spectral=(eV, phi)\] and perform PCA or
> another analysis of the spectral features over different spatial
> observations.
> 
> If our data was called *f*, this can be accomplished with:
> 
> `` ` transformed, decomp =
> decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'],
> PCA) transformed.dims # -> [X, Y, components]``\`
> 
> The results of *decomposition\_along* can be explored with
> *arpes.widgets.pca\_explorer*, regardless of the decomposition class.
> 
>   - Parameters
>     
>       -   - **data** – Input data, can be N-dimensional but should
>             only  
>             include one “spectral” axis.
>     
>       -   - **axes** – Several axes to be treated as a single axis  
>             labeling the list of observations.
>     
>       -   - **decomposition\_cls** – A sklearn.decomposition class
>             (such  
>             as PCA or ICA) to be used to perform the decomposition.
>     
>       -   - **correlation** – Controls whether StandardScaler() is
>             used  
>             as the first stage of the data ingestion pipeline for
>             sklearn.
>     
>       - **kwargs** –
> 
>   - Returns

**arpes.analysis.decomposition.ica\_along(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
axes: List\[str\], decomposition\_cls, correlation=False,**kwargs)\*\*

> Performs a change of basis of your data according to *sklearn*
> decomposition classes. This allows for robust and simple PCA, ICA,
> factor analysis, and other decompositions of your data even when it is
> very high dimensional.
> 
> Generally speaking, PCA and similar techniques work when data is 2D,
> i.e. a sequence of 1D observations. We can make the same techniques
> work by unravelling a ND dataset into 1D (i.e. np.ndarray.ravel()) and
> unravelling a KD set of observations into a 1D set of observations.
> This is basically grouping axes. As an example, if you had a 4D
> dataset which consisted of 2D-scanning valence band ARPES, then the
> dimensions on our dataset would be “\[x,y,eV,phi\]”. We can group
> these into \[spatial=(x, y), spectral=(eV, phi)\] and perform PCA or
> another analysis of the spectral features over different spatial
> observations.
> 
> If our data was called *f*, this can be accomplished with:
> 
> `` ` transformed, decomp =
> decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'],
> PCA) transformed.dims # -> [X, Y, components]``\`
> 
> The results of *decomposition\_along* can be explored with
> *arpes.widgets.pca\_explorer*, regardless of the decomposition class.
> 
>   - Parameters
>     
>       -   - **data** – Input data, can be N-dimensional but should
>             only  
>             include one “spectral” axis.
>     
>       -   - **axes** – Several axes to be treated as a single axis  
>             labeling the list of observations.
>     
>       -   - **decomposition\_cls** – A sklearn.decomposition class
>             (such  
>             as PCA or ICA) to be used to perform the decomposition.
>     
>       -   - **correlation** – Controls whether StandardScaler() is
>             used  
>             as the first stage of the data ingestion pipeline for
>             sklearn.
>     
>       - **kwargs** –
> 
>   - Returns

**arpes.analysis.decomposition.factor\_analysis\_along(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
axes: List\[str\], decomposition\_cls, correlation=False,**kwargs)\*\*

> Performs a change of basis of your data according to *sklearn*
> decomposition classes. This allows for robust and simple PCA, ICA,
> factor analysis, and other decompositions of your data even when it is
> very high dimensional.
> 
> Generally speaking, PCA and similar techniques work when data is 2D,
> i.e. a sequence of 1D observations. We can make the same techniques
> work by unravelling a ND dataset into 1D (i.e. np.ndarray.ravel()) and
> unravelling a KD set of observations into a 1D set of observations.
> This is basically grouping axes. As an example, if you had a 4D
> dataset which consisted of 2D-scanning valence band ARPES, then the
> dimensions on our dataset would be “\[x,y,eV,phi\]”. We can group
> these into \[spatial=(x, y), spectral=(eV, phi)\] and perform PCA or
> another analysis of the spectral features over different spatial
> observations.
> 
> If our data was called *f*, this can be accomplished with:
> 
> `` ` transformed, decomp =
> decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'],
> PCA) transformed.dims # -> [X, Y, components]``\`
> 
> The results of *decomposition\_along* can be explored with
> *arpes.widgets.pca\_explorer*, regardless of the decomposition class.
> 
>   - Parameters
>     
>       -   - **data** – Input data, can be N-dimensional but should
>             only  
>             include one “spectral” axis.
>     
>       -   - **axes** – Several axes to be treated as a single axis  
>             labeling the list of observations.
>     
>       -   - **decomposition\_cls** – A sklearn.decomposition class
>             (such  
>             as PCA or ICA) to be used to perform the decomposition.
>     
>       -   - **correlation** – Controls whether StandardScaler() is
>             used  
>             as the first stage of the data ingestion pipeline for
>             sklearn.
>     
>       - **kwargs** –
> 
>   - Returns

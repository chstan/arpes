# arpes.pipeline module

Although it is not preferred, we support building data analysis
pipelines.

As an analogy, this is kind of the “dual” in the sense of vector space
duality: rather than starting with data and doing things with it to
perform an analysis, you can specify how to chain and apply operations
to the analysis functions and build a pipeline or sequence of operations
that can in the end be applied to data.

This has some distinct advantages:

1.    - You can cache computations before expensive steps, and restart  
        calculations part way through. In fact, this is already
        supported in PyARPES. For instance, you can specify a pipline
        that does
        
        If you run data through this pipeline and have to stop during
        step ii., the next time you run the pipeline, it will start with
        the cached result from step i. instead of recomputing this
        value.

2.  Systematizing certain kinds of analysis

The core of this is *compose* which takes two pipelines (including
atomic elements like single functions) and returns their composition,
which can be paused and restarted between the two parts. Atomic elements
can be constructed with *pipeline*.

In practice though, much of ARPES analysis occurs at scales too small to
make this useful, and interactivity tends to be much preferred to
rigidity. PyARPES nevertheless offers this as an option, as well as
trying to provide support for reproducible and understandable scientific
analyses without sacrificing interativity and a tight feedback loop for
the experimenter.

**exception arpes.pipeline.PipelineRollbackException**

> Bases: `Exception`

**arpes.pipeline.cache\_computation(key, data)**

**arpes.pipeline.compose(\*pipelines)**

**arpes.pipeline.computation\_hash(pipeline\_name, data, intern\_kwargs,
\*args,**kwargs)\*\*

**arpes.pipeline.denormalize\_data(data)**

**arpes.pipeline.normalize\_data(data)**

**arpes.pipeline.pipeline(pipeline\_name=None, intern\_kwargs=None)**

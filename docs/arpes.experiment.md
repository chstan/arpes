# arpes.experiment module

This module is a work-in-progress. Ideally, we would one day like to
offer a simple graphical utility that can communicate with various ARPES
DAQ hardware using a file based interchange format. Currently, DAQ
sequences roughly consistent with the capabilities of the MAESTRO
beamlines are supported via JSON.

More capabiities are also available for Tr-ARPES (shuffling to prevent
laser and drift skewing datasets) with more to come.

**class arpes.experiment.JSONExperimentDriver(queue\_location=None)**

> Bases: `arpes.experiment.ExperimentDriver`
> 
> **dumps(o, desired\_total\_time=None)**

`arpes.experiment.linspace`

> alias of `arpes.experiment.Linspace`

`arpes.experiment.shuffled`

> alias of `arpes.experiment.Shuffled`

`arpes.experiment.move`

> alias of `arpes.experiment.Move`

`arpes.experiment.comment`

> alias of `arpes.experiment.Comment`

`arpes.experiment.collect`

> alias of `arpes.experiment.Collect`

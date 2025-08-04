# Data and inference directories

Pre-computed lensing grids are stored under ``data/tables/<sim_id>/``.
Results of an inference run are written to ``inference/<sim_id>/<run_id>/``.
Each run produces two bookkeeping files:

* ``params.json`` – the run configuration;
* ``metadata.json`` – timestamps, the git commit hash, the ``sim_id`` and the
  pre-computed table version.

Old results can be removed by deleting their directories, e.g.
``rm -r inference/<sim_id>/<run_id>``.

## Reproducing an inference run

1. Read ``git_commit``, ``sim_id`` and ``precomputed_table_version`` from
   ``inference/<sim_id>/<run_id>/metadata.json``.
2. Checkout the recorded commit: ``git checkout <git_commit>``.
3. Ensure the matching pre-computed tables exist under
   ``data/tables/<sim_id>/``.
4. Run ``run_mcmc`` with the parameters stored in ``params.json``.

# It has been extended version

# wait for exec end of Aeta

# now care about the mmdist


# target

- use right Msps relation
- see why multi peak (even small)

---------------
# modify





- mass_sampler.py      &#x2714;
- lens_model.py        &#x2714;
- lens_solver.py       &#x2714;
- lens_properties.py   &#x2714;
- mock_generator.py    &#x2714;
-------------
- cached_A.py          &#x2714;
- interpolator.py      
- utils.py            &#x2714;
----------------------
- likelihood.py
- run_mcmc.py
- main.py


over

# Next step：
01. ~~fix error because of __init__~~
0. ~~consistent model---new branch~~
1. ~~check if all sps error done    ----？~~
2. think about all likelihood use tabulate
3. ~~directly fit the source data~~
4. extended model
5. varid source
6. undetected source
7. 0 In 4D parameter space, perform large-scale random sampling first, then iteratively select subsets to build the interpolator and expand as needed to balance accuracy and efficiency.

7. microlensing




# another line

- why result on jupyter still have multi peak?


# attention:

know no relation between Mh and Re in the data generated

# Pipeline to generate all magnification distributions

## generate samples ($\kappa$,$\gamma$,$s$)

1. a set, oversampled
2. select part of it

## make enough maps fo every sample

## make distribution

## interpolation





1. IMF in the project ??????
2. the bias


# logMh to correct

# use cython

# better 先验 、 重载先验方法
import numpy as np
import emcee
from emcee import moves
from emcee.backends import HDFBackend
import multiprocessing
from functools import partial
from pathlib import Path
from datetime import datetime
import subprocess
import json
import hashlib
from .likelihood import (
    log_posterior,
    initializer_for_pool,
    load_precomputed_tables,
)  # ⬅ 你已经写了它！
from .utils.io import write_metadata, read_metadata

def run_mcmc(
    data_df,
    sim_id,
    log_posterior_func,
    backend_file="chains_eta.h5",
    nwalkers=50,
    nsteps=3000,
    ndim=5,
    initial_guess=None,
    resume=True,
    processes=None,
    use_advanced_moves=True,
):
    """Run MCMC sampling for the lens model parameters.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Observed data for each lens system.
    sim_id : str
        Identifier for pre-computed lensing tables.
    log_posterior_func : callable
        Function computing the log-posterior.
    backend_file : str, optional
        Name of the HDF5 backend file.
    nwalkers, nsteps, ndim : int, optional
        Sampler configuration parameters.
    initial_guess : array-like, optional
        Starting point for walkers when no previous chain exists.
    resume : bool, optional
        Resume from existing backend if available.
    processes : int, optional
        Number of worker processes for parallelization.
    use_advanced_moves : bool, optional
        If True, apply a differential evolution proposal to enhance
        sampling efficiency.

    Returns
    -------
    emcee.EnsembleSampler
        The configured and executed sampler instance.
    """
    if processes is None:
        processes = max(1, int(multiprocessing.cpu_count() // 1.5))

    # --- bookkeeping and directory structure ---
    commit_hash = "unknown"
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        pass

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "inference" / sim_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_path = out_dir / backend_file

    params = {
        "backend_file": backend_file,
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "ndim": int(ndim),
        "resume": bool(resume),
        "processes": int(processes),
    }

    tables_meta_path = (
        Path(__file__).resolve().parent / "data" / "tables" / sim_id / "metadata.json"
    )
    table_version = "unknown"
    if tables_meta_path.exists():
        tables_meta = read_metadata(tables_meta_path)
        table_version = hashlib.sha1(
            json.dumps(tables_meta, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]

    start_time = datetime.utcnow().isoformat()
    metadata = {
        "start_time": start_time,
        "git_commit": commit_hash,
        "sim_id": sim_id,
        "precomputed_table_version": table_version,
    }
    write_metadata(out_dir / "params.json", params)
    write_metadata(out_dir / "metadata.json", metadata)

    tables = load_precomputed_tables(sim_id, required_count=len(data_df))
    if len(tables) < len(data_df):
        raise ValueError(
            f"Not enough precomputed tables for sim_id '{sim_id}'"
        )

    if resume and backend_path.exists():
        print(f"[INFO] 继续采样：读取已有文件 {backend_path}")
        backend = HDFBackend(backend_path, read_only=False)
    else:
        print(f"[INFO] 新建采样：创建新文件 {backend_path}")
        backend = HDFBackend(backend_path)
        backend.reset(nwalkers, ndim)

    if backend.iteration == 0:
        assert initial_guess is not None, "初始值必须提供"
        print("[INFO] 从头开始采样")
        p0 = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
    else:
        print(f"[INFO] 从第 {backend.iteration} 步继续采样")
        p0 = None

    # ✅ 只传入 eta 参数，数据由 initializer 设置为每个子进程的全局变量
    logpost = partial(log_posterior_func)

    move = None
    if use_advanced_moves:
        move = moves.DEMove()

    with multiprocessing.get_context("spawn").Pool(
        processes=processes,
        initializer=initializer_for_pool,
        initargs=(data_df, tables),
    ) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost, pool=pool, backend=backend, moves=move
        )
        sampler.run_mcmc(p0, nsteps, progress=True)



    metadata["end_time"] = datetime.utcnow().isoformat()
    write_metadata(out_dir / "metadata.json", metadata)

    print("[INFO] 采样完成")
    return sampler

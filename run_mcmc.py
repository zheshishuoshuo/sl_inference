import numpy as np
import emcee
from emcee.backends import HDFBackend
import multiprocessing
from functools import partial
from pathlib import Path
from datetime import datetime
from .likelihood import (
    log_posterior,
    initializer_for_pool,
    load_precomputed_tables,
)  # ⬅ 你已经写了它！
from .utils.io import write_metadata

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
):
    if processes is None:
        processes = max(1, int(multiprocessing.cpu_count() // 1.5))

    out_dir = Path(__file__).resolve().parent / "data" / "chains" / sim_id
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
    start_time = datetime.utcnow().isoformat()
    write_metadata(out_dir / "params.json", params)
    write_metadata(out_dir / "metadata.json", {"start_time": start_time})

    tables = load_precomputed_tables(sim_id)
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

    with multiprocessing.get_context("spawn").Pool(
        processes=processes,
        initializer=initializer_for_pool,
        initargs=(data_df, tables),
    ) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost, pool=pool, backend=backend
        )
        sampler.run_mcmc(p0, nsteps, progress=True)



    write_metadata(
        out_dir / "metadata.json",
        {"start_time": start_time, "end_time": datetime.utcnow().isoformat()},
    )

    print("[INFO] 采样完成")
    return sampler

import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from joblib import Parallel, delayed

import os
from concurrent.futures import ProcessPoolExecutor, as_completed



import h5py
import numpy as np
from tqdm.auto import tqdm

import ronswanson.grids as grids
from ronswanson.utils.color import Colors
from ronswanson.utils.configuration import ronswanson_config

from .utils.file_open import open_component_file
from .utils.logging import setup_logger

log = setup_logger(__name__)

_STORE_RE = re.compile(r"^sim_store_(\d+)\.h5$")

class Timer(object):
    """
    A timing context manager
    """

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.total_time = time.perf_counter() - self.start_time


class Simulation(metaclass=ABCMeta):
    def __init__(
        self,
        simulation_id: int,
        parameter_set: Dict[str, float],
        energy_grid: List[grids.EnergyGrid],
        out_file: str,
        num_outputs: int = 1,
        num_meta_parameters: Optional[int] = None,
    ) -> None:

        """
        Generic simulation class

        :param simulation_id:
        :type simulation_id: int
        :param parameter_set:
        :type parameter_set: Dict[str, float]
        :param energy_grid:
        :type energy_grid: np.ndarray
        :param out_file:
        :type out_file: str
        :returns:
        """
        self._out_file: str = out_file
        self._parameter_set: Dict[str, float] = parameter_set
        self._simulation_id: int = simulation_id
        self._energy_grid: List[grids.EnergyGrid] = energy_grid
        self._num_outputs: int = num_outputs
        self._num_meta_parameters: Optional[int] = num_meta_parameters

        if not len(self._energy_grid) == self._num_outputs:

            log.error(f"requested number of outputs {self._num_outputs}")
            log.error(f"but only have {len(self._energy_grid)} energy grids")

            raise RuntimeError()

    def run(self) -> None:

        """
        run this simulation
        :returns:

        """

        # run the simulation

        log.debug(f"simulation {self._simulation_id} is now running")

        with Timer() as timer:

            output: Dict[str, np.ndarray] = self._run_call()

        with open_component_file(self._out_file, self._simulation_id) as f:

            f.attrs['run_time'] = timer.total_time

            f.create_dataset(
                "parameters", data=np.array(list(self._parameter_set.values()))
            )

            for i in range(self._num_outputs):

                f.create_dataset(f"output_{i}", data=output[f"output_{i}"])

            if self._num_meta_parameters is not None:

                for i in range(self._num_meta_parameters):

                    f.attrs[f'meta_{i}'] = output[f"meta_{i}"]

    @abstractmethod
    def _run_call(self) -> Dict[str, np.ndarray]:

        log.error("Attempting to use base class")

        raise RuntimeError()


def _get_sim_id(path: Path) -> int:
    match = _STORE_RE.match(path.name)

    if match is None:
        raise ValueError(f"Unexpected store filename: {path}")

    return int(match.group(1))


def _start_worker() -> int:
    """Force worker creation before the parent opens the output HDF5 file."""
    return os.getpid()


def _read_store(path: str, value_keys: tuple, meta_keys: tuple):
    """Read one small source HDF5 file."""
    store = Path(path)
    sim_id = _get_sim_id(store)

    with h5py.File(path, "r") as f:
        return (
            sim_id,
            f["parameters"][()],
            tuple(f[key][()] for key in value_keys),
            f.attrs["run_time"],
            tuple(f.attrs[key] for key in meta_keys),
        )


def _read_store_group(paths: list[str], value_keys: tuple, meta_keys: tuple):
    """
    Read many files in one pool task.

    This is important for ~3.8 kB files: submitting one task per file adds
    excessive scheduling and IPC overhead.
    """
    return [_read_store(path, value_keys, meta_keys) for path in paths]


def _delete_store_group(paths: list[str]):
    """Delete a group of source files after gathering completed safely."""
    errors = []

    for path in paths:
        try:
            Path(path).unlink()
        except Exception as error:
            errors.append((path, str(error)))

    return errors


def _contiguous_runs(indices: np.ndarray):
    """Yield positions corresponding to consecutive destination rows."""
    if len(indices) == 0:
        return

    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(indices)]))

    for start, stop in zip(starts, stops):
        yield int(start), int(stop)


def _make_write_array(items, dataset):
    """Create an appropriate NumPy array for an HDF5 dataset."""
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return np.asarray(items, dtype=object)

    if h5py.check_vlen_dtype(dataset.dtype) is not None:
        result = np.empty(len(items), dtype=object)
        result[:] = items
        return result

    return np.asarray(items)


def _write_runs(dataset, indices: np.ndarray, values) -> None:
    """Write contiguous groups as slices instead of one HDF5 write per row."""
    for start, stop in _contiguous_runs(indices):
        destination = slice(int(indices[start]), int(indices[stop - 1]) + 1)
        dataset[destination] = values[start:stop]


def gather(
    file_name: str,
    current_size: int = 0,
    clean: bool = True,
    *,
    workers: int | None = None,
    batch_size: int = 16384,
    task_size: int = 512,
) -> None:
    """
    Gather multi-file simulation results into one HDF5 file.

    Defaults are tuned for many very small (~kB) HDF5 files on NFS.

    Parameters
    ----------
    file_name : str
        Destination database filename.
    current_size : int
        Offset into the destination datasets.
    clean : bool
        Remove individual source files after gathering.
    workers : int | None
        Number of parallel readers. Defaults to min(16, CPU count).
    batch_size : int
        Number of files buffered and written per output batch.
    task_size : int
        Number of tiny source files read by each process-pool task.
    """

    if workers is None:
        # NFS metadata servers usually saturate well before 64+ workers.
        workers = min(16, os.cpu_count() or 1)

    workers = max(1, int(workers))
    batch_size = max(1, int(batch_size))
    task_size = max(1, int(task_size))

    p = Path(file_name)

    if ronswanson_config.slurm.store_dir is None:
        parent_dir = p.absolute().parent
    else:
        parent_dir = Path(ronswanson_config.slurm.store_dir).absolute()

    multi_file_dir = parent_dir / f"{p.stem}_store"

    files = sorted(
        multi_file_dir.glob("sim_store_*.h5"),
        key=_get_sim_id,
    )

    if not files:
        log.warning(f"No simulation store files found in {multi_file_dir}")
        return

    # Start workers before this process opens any HDF5 files.
    # This avoids unsafe fork-after-HDF5-open behavior on Linux.
    with ProcessPoolExecutor(max_workers=workers) as executor:

        warmup = [executor.submit(_start_worker) for _ in range(workers)]
        for future in warmup:
            future.result()

        # Discover the destination schema once.
        with h5py.File(file_name, "a") as database_file:
            value_keys = tuple(database_file["values"].keys())
            meta_keys = (
                tuple(database_file["meta"].keys())
                if "meta" in database_file
                else ()
            )

        # Keep output writes serial: standard HDF5 does not permit concurrent
        # writers safely. Reads are parallel; writes are bulk contiguous slices.
        with h5py.File(file_name, "a") as database_file:

            database_file.attrs["has_been_touched"] = True

            parameters_ds = database_file["parameters"]
            run_time_ds = database_file["run_time"]

            values_ds = {
                key: database_file["values"][key]
                for key in value_keys
            }

            meta_ds = {
                key: database_file["meta"][key]
                for key in meta_keys
            }

            with tqdm(
                total=len(files),
                desc="Gathering simulations",
                colour=Colors.GREEN.value,
            ) as progress:

                for batch_start in range(0, len(files), batch_size):
                    batch = files[batch_start : batch_start + batch_size]

                    # Submit groups of files, not individual 3.8 kB files.
                    groups = [
                        [str(path) for path in batch[i : i + task_size]]
                        for i in range(0, len(batch), task_size)
                    ]

                    futures = {
                        executor.submit(
                            _read_store_group,
                            group,
                            value_keys,
                            meta_keys,
                        ): len(group)
                        for group in groups
                    }

                    results = []

                    for future in as_completed(futures):
                        results.extend(future.result())
                        progress.update(futures[future])

                    # Futures complete in arbitrary order; restore sequential
                    # write order to minimize HDF5 chunk/cache thrashing.
                    results.sort(key=lambda result: result[0])

                    indices = np.asarray(
                        [result[0] + current_size for result in results],
                        dtype=np.intp,
                    )

                    parameters = _make_write_array(
                        [result[1] for result in results],
                        parameters_ds,
                    )

                    run_times = _make_write_array(
                        [result[3] for result in results],
                        run_time_ds,
                    )

                    values = {
                        key: _make_write_array(
                            [result[2][i] for result in results],
                            values_ds[key],
                        )
                        for i, key in enumerate(value_keys)
                    }

                    meta = {
                        key: _make_write_array(
                            [result[4][i] for result in results],
                            meta_ds[key],
                        )
                        for i, key in enumerate(meta_keys)
                    }

                    _write_runs(parameters_ds, indices, parameters)
                    _write_runs(run_time_ds, indices, run_times)

                    for key in value_keys:
                        _write_runs(values_ds[key], indices, values[key])

                    for key in meta_keys:
                        _write_runs(meta_ds[key], indices, meta[key])

        # Do deletion only after the output file was successfully closed.
        if clean:
            delete_groups = [
                [str(path) for path in files[i : i + task_size]]
                for i in range(0, len(files), task_size)
            ]

            delete_futures = [
                executor.submit(_delete_store_group, group)
                for group in delete_groups
            ]

            delete_errors = []

            for future in tqdm(
                as_completed(delete_futures),
                total=len(delete_futures),
                desc="Removing simulation stores",
                colour=Colors.GREEN.value,
            ):
                delete_errors.extend(future.result())

            if delete_errors:
                example_path, example_error = delete_errors[0]
                raise RuntimeError(
                    f"Could not remove {len(delete_errors)} store files. "
                    f"Example: {example_path}: {example_error}"
                )

    if clean:
        multi_file_dir.rmdir()

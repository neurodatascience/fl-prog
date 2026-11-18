from collections.abc import Iterable

import numpy as np

from fl_prog.utils.math import multivariate_logistic
from fl_prog.utils.validation import check_rng


@check_rng
def generate_timepoints(
    n_timepoints: int,
    t0_min: float = 0.0,
    t0_max: float = 1.0,
    acceleration_factor: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate timepoints for a subject.

    Parameters
    ----------
    n_timepoints : int
    t0_min : float, optional
        Minimum value for the first timepoint, by default 0.0
    t0_max : float, optional
        Maximum value for the first timepoint, by default 1.0
    acceleration_factor : float, optional
        Acceleration factor for the subject, by default 1.0
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        Shape (n_timepoints,)
    """
    t0_for_subject = rng.uniform(t0_min, t0_max)

    # start from t0 then uniform to t0_max
    timepoints = np.concatenate(
        (
            [0],
            rng.uniform(0, t0_max - t0_for_subject, size=n_timepoints - 1)
            * acceleration_factor,
        )
    )
    return np.sort(timepoints) + t0_for_subject


@check_rng
def simulate_all_subjects(
    n_subjects: int,
    k_values: Iterable[float],
    x0_values: Iterable[float],
    scaling_factors: Iterable[float],
    sigmas: Iterable[float],
    max_n_timepoints: int = 3,
    n_timepoints_distribution: dict[str | int, float] | None = None,
    time_at_timepoint: list[float] | None = None,
    shift_time: bool = False,
    t0_min: float = 0.0,
    t0_max: float = 1.0,
    acceleration_factor_std: float = 0.0,
    acceleration_factor_mean: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Simulate timepoints and biomarkers for all subjects.

    Subjects can have multiple timepoints, but they are concatenated into a single
    dimension in the output arrays.

    Parameters
    ----------
    n_subjects : int
    k_values : Iterable[float]
        Logistic function steepness parameters.
    x0_values : Iterable[float]
        Logistic function midpoint parameters. Must have the same length as k_values.
    scaling_factors : Iterable[float]
        Scaling factors for each biomarker. Must have the same length as k_values.
    sigmas : Iterable[float]
        Standard deviations for Gaussian noise added to biomarkers. Must have the same length as k_values.
    max_n_timepoints : int, optional
        Maximum number of timepoints for each subject. Ignored if n_timepoints_distribution and time_at_timepoint are provided.
    n_timepoints_distribution : dict[str | int, float] | None, optional
        A dictionary specifying the distribution of the number of timepoints per subject. time_at_timepoint must also be provided.
    time_at_timepoint : list[float] | None, optional
        A list of time values corresponding to each timepoint index. n_timepoints_distribution must also be provided.
    shift_time : bool, optional
        If True, shift timepoints so that the first timepoint for each subject is at 0.
    t0_min : float, optional
        Minimum value for the first timepoint of each subject.
    t0_max : float, optional
        Maximum value for the first timepoint of each subject.
    acceleration_factor_std : float, optional
        Standard deviation for the acceleration factor of each subject.
    acceleration_factor_mean : float, optional
        Mean of the normal distribution from which the acceleration factor of each
        subject is sampled, by default 1.0.
    rng : np.random.Generator, optional

    Returns
    -------
    Tuple[np.ndarray, list[np.ndarray], np.ndarray]
        Timepoints: (n_total_timepoints,)
        Biomarkers: list of length n_subjects with elements of shape (n_timepoints,
                    n_biomarkers),
        Time shift values: (n_subjects,)
    """
    if (
        len(k_values) != len(x0_values)
        or len(k_values) != len(scaling_factors)
        or len(k_values) != len(sigmas)
    ):
        raise ValueError(
            "k_values, x0_values, scaling_factors, and sigmas must "
            f"have the same length. Got {len(k_values)}, {len(x0_values)}, "
            f"{len(scaling_factors)}, and {len(sigmas)}."
        )

    if (n_timepoints_distribution is not None and time_at_timepoint is None) or (
        n_timepoints_distribution is None and time_at_timepoint is not None
    ):
        raise ValueError(
            "n_timepoints_distribution and time_at_timepoint must both be provided."
        )

    timepoints_all = []
    biomarkers_all = []
    acceleration_factors = []
    for _ in range(n_subjects):
        acceleration_factor = -1
        while acceleration_factor <= 0:
            # will be reset to 1 if n_timepoints is 1
            acceleration_factor = rng.normal(
                acceleration_factor_mean, acceleration_factor_std
            )
        if n_timepoints_distribution is not None:
            n_timepoints = int(
                rng.choice(
                    list(n_timepoints_distribution.keys()),
                    p=(
                        np.asarray(
                            list(n_timepoints_distribution.values()), dtype=float
                        )
                        / sum(n_timepoints_distribution.values())
                    ),
                )
            )
            if n_timepoints == 1:
                acceleration_factor = 1.0
            t0_for_subject = rng.uniform(t0_min, t0_max)
            timepoints = (
                np.array(time_at_timepoint)[:n_timepoints] * acceleration_factor
                + t0_for_subject
            )
        else:
            n_timepoints = rng.integers(1, max_n_timepoints + 1)
            if n_timepoints == 1:
                acceleration_factor = 1.0
            timepoints = generate_timepoints(
                n_timepoints,
                t0_min=t0_min,
                t0_max=t0_max,
                acceleration_factor=acceleration_factor,
                rng=rng,
            )

        biomarkers = multivariate_logistic(
            timepoints,
            k_values,
            x0_values,
            scaling_factors,
        )
        for i_biomarker, sigma in enumerate(sigmas):
            biomarkers[:, i_biomarker] += rng.normal(0, sigma, size=biomarkers.shape[0])

        timepoints_all.append(timepoints)
        biomarkers_all.append(biomarkers)
        acceleration_factors.append(acceleration_factor)

    if shift_time:
        # shift timepoints so that the first timepoint is at 0
        time_shifts = [tp[0] for tp in timepoints_all]
        timepoints_all = [
            (tp - tp[0]) / acceleration_factor
            for tp, acceleration_factor in zip(timepoints_all, acceleration_factors)
        ]
    else:
        time_shifts = [0.0] * n_subjects

    return (
        timepoints_all,
        biomarkers_all,
        np.array(time_shifts),
        np.array(acceleration_factors),
    )


def get_overlap_windows(n_sites: int, overlap: float) -> list[tuple[float, float]]:
    """Get equally spaced windows on [0, 1] where neighbours share a fraction of width.

    Examples
    --------
    >>> get_overlap_windows(4, 0)
    [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    >>> [(round(a, 2), round(b, 2)) for a, b in get_overlap_windows(4, 0.5)]
    [(0.0, 0.4), (0.2, 0.6), (0.4, 0.8), (0.6, 1.0)]
    >>> get_overlap_windows(4, 1)
    [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if not 0 <= overlap <= 1:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}")
    width = 1 / (1 + (n_sites - 1) * (1 - overlap))
    step = width * (1 - overlap)
    return [(i * step, i * step + width) for i in range(n_sites)]


@check_rng
def assign_sites_by_overlap(
    t0: Iterable[float],
    n_sites: int,
    overlap: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Assign subjects to sites based on their t0 (disease stage at first visit).

    1. Subjects are ranked by t0. Each subject gets a position ``u = rank / N`` in [0, 1).
       Ties are broken randomly, so the result doesn't depend on input order. Using
       ranks rather than raw t0 means the windows split subjects evenly, however t0
       is distributed.
    2. Each site is given a window on u, using ``get_overlap_windows(n_sites, overlap)``.
    3. Sites are filled in in order. Each site has a capacity of ``N / n_sites`` (±1).
       Site i:
       - First takes every unassigned subject left of the next site's window
         (``u < next_start``). No later site could take them, so site i must.
       - Then fills its remaining places at random from the unassigned subjects in
         the part of its window that overlaps the next site
         (``next_start <= u <= end``).
       - The last site takes everyone left.

    ``overlap=0`` gives contiguous t0 chunks, ``overlap=1`` gives a random split.

    Returns
    -------
    np.ndarray
        Site index (0 to n_sites - 1) for each subject, shape (N,).
    """
    t0 = np.asarray(t0, dtype=float)
    n_subjects = len(t0)
    if n_subjects < n_sites:
        raise ValueError(f"Need at least {n_sites} subjects, got {n_subjects}")

    # break ties randomly so that the assignment does not depend on input order
    order = np.lexsort((rng.random(n_subjects), t0))
    u = np.empty(n_subjects)
    u[order] = np.arange(n_subjects) / n_subjects

    windows = get_overlap_windows(n_sites, overlap)
    capacities = np.full(n_sites, n_subjects // n_sites)
    capacities[: n_subjects % n_sites] += 1

    sites = np.full(n_subjects, -1, dtype=int)
    eps = 1e-12  # tolerance for float window edges
    for i_site, (start, end) in enumerate(windows):
        unassigned = sites == -1
        if i_site == n_sites - 1:
            sites[unassigned] = i_site
            break
        next_start = windows[i_site + 1][0]
        forced = np.flatnonzero(unassigned & (u < next_start - eps))
        n_remaining = capacities[i_site] - len(forced)
        if n_remaining < 0:
            raise RuntimeError(
                f"Site {i_site} must take {len(forced)} subjects, "
                f"more than its capacity ({capacities[i_site]})"
            )
        candidates = np.flatnonzero(
            unassigned & (u >= next_start - eps) & (u < end + eps)
        )
        if len(candidates) < n_remaining:
            raise RuntimeError(
                f"Site {i_site} needs {n_remaining} more subjects but only "
                f"{len(candidates)} are available in its window"
            )
        chosen = rng.choice(candidates, size=n_remaining, replace=False)
        sites[forced] = i_site
        sites[chosen] = i_site

    return sites

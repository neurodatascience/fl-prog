from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

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


Window = tuple[float, float]


def _rank_positions(t0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Get the position ``rank / N`` in [0, 1) of each subject when ranked by t0.

    Ties are broken randomly, so the result doesn't depend on input order.
    """
    n_subjects = len(t0)
    order = np.lexsort((rng.random(n_subjects), t0))
    u = np.empty(n_subjects)
    u[order] = np.arange(n_subjects) / n_subjects
    return u


@check_rng
def assign_sites_by_windows(
    t0: Iterable[float],
    windows_per_site: Sequence[Sequence[Window]],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Assign subjects to sites based on their t0 (disease stage at first visit).

    1. Subjects are ranked by t0. Each subject gets a position ``u = rank / N`` in
       [0, 1). Ties are broken randomly. Using ranks rather than raw t0 means the
       windows split subjects evenly, however t0 is distributed.
    2. Each site has one or more windows on u, e.g. ``[(0, 1)]`` for full coverage
       or ``[(0, 0.25), (0.75, 1)]`` for only early and late subjects. A subject can
       go to a site if its u is in one of the site's windows.
    3. Each site has a capacity of ``N / n_sites`` (±1). Subjects are matched to site
       places at random among the valid assignments (linear assignment with random
       costs), so a subject in the windows of several sites can go to any of them.

    Returns
    -------
    np.ndarray
        Site index (0 to n_sites - 1) for each subject, shape (N,).
    """
    t0 = np.asarray(t0, dtype=float)
    n_subjects = len(t0)
    n_sites = len(windows_per_site)
    if n_sites < 1:
        raise ValueError("Need at least 1 site")
    if n_subjects < n_sites:
        raise ValueError(f"Need at least {n_sites} subjects, got {n_subjects}")

    u = _rank_positions(t0, rng)

    capacities = np.full(n_sites, n_subjects // n_sites)
    capacities[: n_subjects % n_sites] += 1

    eps = 1e-12  # tolerance for float window edges
    eligible = np.zeros((n_subjects, n_sites), dtype=bool)
    for i_site, windows in enumerate(windows_per_site):
        for start, end in windows:
            eligible[:, i_site] |= (u >= start - eps) & (u < end + eps)

    # one column per site place
    slot_sites = np.repeat(np.arange(n_sites), capacities)
    eligible_slots = eligible[:, slot_sites]
    cost = np.where(eligible_slots, rng.random(eligible_slots.shape), n_subjects + 1.0)
    i_subjects, i_slots = linear_sum_assignment(cost)

    if not eligible_slots[i_subjects, i_slots].all():
        raise RuntimeError(
            "Cannot assign every subject to a site with these windows "
            f"(site capacities: {capacities.tolist()}, subjects in each site's "
            f"windows: {eligible.sum(axis=0).tolist()}, subjects in no window: "
            f"{int((~eligible.any(axis=1)).sum())})"
        )

    sites = np.empty(n_subjects, dtype=int)
    sites[i_subjects] = slot_sites[i_slots]
    return sites

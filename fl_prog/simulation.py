import numpy as np
from typing import Iterable, Optional, Tuple

from fl_prog.utils.math import multivariate_logistic
from fl_prog.utils.validation import check_rng


@check_rng
def generate_timepoints(
    n_timepoints: int,
    t0_min: float = 0.0,
    t0_max: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate timepoints for a subject.

    Parameters
    ----------
    n_timepoints : int
    t0_min : float, optional
        Minimum value for the first timepoint, by default 0.0
    t0_max : float, optional
        Maximum value for the first timepoint, by default 1.0
    rng : Optional[np.random.Generator], optional

    Returns
    -------
    np.ndarray
        Shape (n_timepoints,)
    """
    t0_for_subject = rng.uniform(t0_min, t0_max)

    # start from t0 then uniform to t0_max
    timepoints = np.concatenate(
        (
            [t0_for_subject],
            rng.uniform(t0_for_subject, t0_max, size=n_timepoints - 1),
        )
    )
    return np.sort(timepoints)


@check_rng
def simulate_all_subjects(
    n_subjects: int,
    k_values: Iterable[float],
    x0_values: Iterable[float],
    vertical_shifts: Iterable[float],
    scaling_factors: Iterable[float],
    sigmas: Iterable[float],
    max_n_timepoints: int = 3,
    n_timepoints_distribution: dict[str | int, float] | None = None,
    time_at_timepoint: list[float] | None = None,
    shift_time: bool = False,
    t0_min: float = 0.0,
    t0_max: float = 1.0,
    acceleration_factor_min: float = 1.0,
    acceleration_factor_max: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
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
    vertical_shifts : Iterable[float]
        Vertical shift parameters for each biomarker. Must have the same length as k_values.
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
    acceleration_factor_min : float, optional
        Minimum value for the acceleration factor of each subject.
    acceleration_factor_max : float, optional
        Maximum value for the acceleration factor of each bsubject.
    rng : Optional[np.random.Generator], optional

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
        or len(k_values) != len(vertical_shifts)
        or len(k_values) != len(scaling_factors)
        or len(k_values) != len(sigmas)
    ):
        raise ValueError(
            "k_values, x0_values, vertical_shifts, scaling_factors, and sigmas must "
            f"have the same length. Got {len(k_values)}, {len(x0_values)}, "
            f"{len(vertical_shifts)}, {len(scaling_factors)}, and {len(sigmas)}."
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
        acceleration_factor = rng.uniform(
            acceleration_factor_min, acceleration_factor_max
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
            t0_for_subject = rng.uniform(t0_min, t0_max)
            timepoints = np.array(time_at_timepoint)[:n_timepoints] + t0_for_subject
        else:
            n_timepoints = rng.integers(1, max_n_timepoints + 1)
            timepoints = generate_timepoints(
                n_timepoints, t0_min=t0_min, t0_max=t0_max, rng=rng
            )

        biomarkers = multivariate_logistic(
            timepoints * acceleration_factor,
            k_values,
            x0_values,
            vertical_shifts,
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
        timepoints_all = [tp - tp[0] for tp in timepoints_all]
    else:
        time_shifts = [0.0] * n_subjects

    return (
        timepoints_all,
        biomarkers_all,
        np.array(time_shifts),
        np.array(acceleration_factors),
    )

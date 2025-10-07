import numpy as np
from typing import Optional, Tuple

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
    max_n_timepoints: int = 3,
    n_biomarkers: int = 5,
    shift_time: bool = False,
    k_min: float = 5.0,
    k_max: float = 10.0,
    x0_min: float = 0,
    x0_max: float = 1.0,
    t0_min: float = 0.0,
    t0_max: float = 1.0,
    sigma: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Simulate timepoints and biomarkers for all subjects.

    Subjects can have multiple timepoints, but they are concatenated into a single
    dimension in the output arrays.

    Parameters
    ----------
    n_subjects : int
    max_n_timepoints : int, optional
    n_biomarkers : int, optional
    shift_time : bool, optional
        If True, shift timepoints so that the first timepoint for each subject is at 0.
    k_min : float, optional
        Minimum value for logistic function steepness parameter
    k_max : float, optional
        Maximum value for logistic function steepness parameter
    x0_min : float, optional
        Minimum value for logistic function midpoint parameter
    x0_max : float, optional
        Maximum value for logistic function midpoint parameter
    t0_min : float, optional
        Minimum value for the first timepoint of each subject
    t0_max : float, optional
        Maximum value for the first timepoint of each subject
    sigma : float, optional
        Standard deviation for Gaussian noise added to biomarkers
    rng : Optional[np.random.Generator], optional

    Returns
    -------
    Tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]
        Timepoints: (n_total_timepoints,)
        Biomarkers: list of length n_subjects with elements of shape (n_timepoints,
                    n_biomarkers),
        Time shift values: (n_subjects,)
        k values: (n_biomarkers,)
        x0 values: (n_biomarkers,)
    """
    k_values = rng.uniform(k_min, k_max, size=n_biomarkers)
    x0_values = rng.uniform(x0_min, x0_max, size=n_biomarkers)

    timepoints_all = []
    biomarkers_all = []
    for _ in range(n_subjects):
        n_timepoints = rng.integers(1, max_n_timepoints + 1)
        timepoints = generate_timepoints(
            n_timepoints, t0_min=t0_min, t0_max=t0_max, rng=rng
        )

        biomarkers = multivariate_logistic(timepoints, k_values, x0_values)
        biomarkers += rng.normal(0, sigma, size=biomarkers.shape)

        timepoints_all.append(timepoints)
        biomarkers_all.append(biomarkers)

    if shift_time:
        # shift timepoints so that the first timepoint is at 0
        time_shifts = [tp[0] for tp in timepoints_all]
        timepoints_all = [tp - tp[0] for tp in timepoints_all]
    else:
        time_shifts = [0.0] * n_subjects

    return (timepoints_all, biomarkers_all, np.array(time_shifts), k_values, x0_values)

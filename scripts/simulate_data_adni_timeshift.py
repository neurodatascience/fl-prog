#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_DATA, load_json, save_json

DEFAULT_ADNI_DIR = Path("data/adni")
DEFAULT_ADNI_CONFIG = DEFAULT_ADNI_DIR / "config.json"
DEFAULT_ADNI_MERGE_CSV = DEFAULT_ADNI_DIR / "ADNIMERGE.csv"
DEFAULT_ADNI_FREESURFER_CSV = DEFAULT_ADNI_DIR / "ADNI_freesurfer7p1p1-forSantiago.csv"

DEFAULT_TAG = "simulated_adni_timeshift"
DEFAULT_RNG_SEED = 42
DEFAULT_VISIT_MONTHS = [0, 6, 12, 24, 36]

FS_COL_SUBJECT = "rid"
FS_COL_VISIT = "visit"
FS_COL_AGE = "Age"
FS_COL_SEX = "Sex"
# FS_COL_MMSE = "MMSE"
FS_COL_DX = "DX"

MERGE_COL_SUBJECT = "RID"
MERGE_COL_VISIT = "VISCODE"
MERGE_COL_AGE = "AGE"
MERGE_COL_SITE = "SITE"
MERGE_COL_DX = "DX"
MERGE_COL_DX_BL = "DX_bl"
ALLOWED_BASELINE_DX = ("CN", "SMC", "EMCI", "LMCI", "AD")

COL_TRUE_ONSET_AGE = "true_onset_age"
COL_TRUE_DISEASE_STAGE = "true_disease_stage"
COL_TRUE_TIME_SHIFT = "true_time_shift"

DEFAULT_LOGISTIC_SLOPE = 0.25
DEFAULT_LOGISTIC_MIDPOINT = (
    0.0  # biomarker changes most rapidly at true disease stage = 0 (onset)
)

# Prior distributions per diagnosis for sampling hidden true time-shift (true onset age) in the simulator.
# These are learned from ADNI data using DPMoSt. Note: in months
#     disease_time_months = visit_month + true_time_shift
#
# true_time_shift is the subject's latent disease-time position at baseline.
PRIORS_TAG = "adni_iid"
PRIORS_SETUP = "centralized"
DEFAULT_TIME_SHIFT_PRIORS_JSON = (
    DEFAULT_ADNI_DIR / f"time_shift_priors_{PRIORS_SETUP}_{PRIORS_TAG}.json"
)
# # Normal fallback distributions used when empirical samples are unavailable.
# Currently, set to values from fitted DPMoSt model on ADNI data.
DX_TIMESHIFT_PARAMS = {  # mean, stddev in months
    "CN": (98.0, 69.9),
    "SMC": (71.4, 61.4),
    "EMCI": (85.9, 69.8),
    "LMCI": (160.5, 80.8),
    "AD": (210.5, 65.4),
}


@dataclass
class BiomarkerParams:
    """Simple learned parameters for one simulated biomarker."""

    name: str
    low: float
    high: float
    slope: float
    midpoint: float
    direction: int
    noise_sd: float
    marginal_mean: float
    marginal_sd: float


@dataclass
class Calibration:
    """Population-level ADNI statistics used by the baseline simulator."""

    n_subjects: int
    visit_months: list[int]
    visit_count_probs: dict[int, float]
    age_mean: float
    age_sd: float
    age_min: float
    age_max: float
    sex_probs: dict[str, float]
    dx_probs: dict[str, float]
    site_probs: dict[str, float]
    biomarkers: dict[str, BiomarkerParams]
    noise_corr: list[list[float]]


def _visit_to_month(value: Any) -> int | None:
    """Convert Freesurfer visits such as M00, M06, M12 into month integers."""
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value.startswith("M"):
        try:
            return int(value[1:])
        except ValueError:
            return None

    if value in {"BL", "SC", "SCMRI"}:
        return 0

    return None


def _month_to_visit(month: int) -> str:
    """Convert integer month to Freesurfer-style visit code."""
    return f"M{month:02d}"


def _month_to_merge_visit(month: int) -> str:
    """Convert integer month to ADNIMERGE-style VISCODE."""
    if month == 0:
        return "bl"
    return f"m{month:02d}"


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric while preserving missing values."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_subject_ids(series: pd.Series) -> pd.Series:
    """Normalize RID values so e.g., 123, '123', and '123.0' all become '123'."""
    result = series.astype("string").str.strip()
    numeric = pd.to_numeric(result, errors="coerce").astype("float64")
    integral = numeric.notna() & numeric.mod(1).eq(0)

    result.loc[integral] = numeric.loc[integral].astype("int64").astype("string")
    return result


def _get_baseline_dx_by_subject(df_merge: pd.DataFrame) -> pd.Series:
    """Return one ADNIMERGE DX_bl value per normalized RID.

    DX_bl is normally repeated across a subject's visits. Baseline rows are
    prioritized if inconsistent values occur.
    """
    diagnosis = df_merge[[MERGE_COL_SUBJECT, MERGE_COL_VISIT, MERGE_COL_DX_BL]].copy()

    diagnosis["subject_key_sim"] = _normalize_subject_ids(diagnosis[MERGE_COL_SUBJECT])
    diagnosis["dx_bl_sim"] = (
        diagnosis[MERGE_COL_DX_BL].astype("string").str.strip().str.upper()
    )

    # Prefer an explicit baseline row when duplicate RID records exist.
    visit_code = diagnosis[MERGE_COL_VISIT].astype("string").str.strip().str.lower()
    diagnosis["baseline_priority_sim"] = np.where(
        visit_code.eq("bl"),
        0,
        np.where(visit_code.isin(["sc", "scmri"]), 1, 2),
    )

    diagnosis = (
        diagnosis.dropna(subset=["subject_key_sim", "dx_bl_sim"])
        .sort_values(["subject_key_sim", "baseline_priority_sim"])
        .drop_duplicates("subject_key_sim", keep="first")
    )

    return diagnosis.set_index("subject_key_sim")["dx_bl_sim"]


def _load_time_shift_priors(
    fpath: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    """Load empirical DPMoSt time-shift samples by diagnosis.

    The returned samples are in months. An empty dictionary means that the
    simulator should use the normal distributions in DX_TIMESHIFT_PARAMS.
    """
    if fpath is None or not fpath.exists():
        print(
            "Time-shift prior file was not found. Using fallback DX_TIMESHIFT_PARAMS."
        )
        return {}, None

    prior_data = load_json(fpath)

    if prior_data.get("units") != "months":
        raise ValueError(
            f"{fpath} must store time shifts in months; "
            f"found {prior_data.get('units')!r}."
        )

    distributions = prior_data.get("distributions_by_dx", {})
    samples_by_dx: dict[str, np.ndarray] = {}

    for diagnosis, distribution in distributions.items():
        samples = np.asarray(distribution.get("samples", []), dtype=float)

        if len(samples) > 0:
            samples_by_dx[str(diagnosis)] = samples

    if not samples_by_dx:
        print(
            "The time-shift prior file contained no usable samples. "
            "Using fallback DX_TIMESHIFT_PARAMS."
        )
        return {}, prior_data

    print(f"Loaded time-shift priors from {fpath}")
    print("Available diagnoses: " + ", ".join(sorted(samples_by_dx)))

    return samples_by_dx, prior_data


def _get_stage_mean_years_by_dx(
    time_shift_samples_by_dx: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return mean baseline disease-time positions in years by diagnosis."""
    stage_means = {
        diagnosis: float(np.mean(samples) / 12.0)
        for diagnosis, samples in time_shift_samples_by_dx.items()
        if len(samples) > 0
    }

    for diagnosis, (mean_months, _sd_months) in DX_TIMESHIFT_PARAMS.items():
        stage_means.setdefault(diagnosis, mean_months / 12.0)

    return stage_means


def _sample_categorical(
    rng: np.random.Generator,
    probs: dict[str, float],
    size: int | None = None,
) -> Any:
    """Sample labels from a probability dictionary."""
    labels = np.array(list(probs.keys()))
    p = np.array(list(probs.values()), dtype=float)
    p = p / p.sum()
    return rng.choice(labels, size=size, p=p)


def _nearest_psd_corr(corr: np.ndarray) -> np.ndarray:
    """Make an empirical correlation matrix (PSD) usable for multivariate noise."""
    corr = np.asarray(corr, dtype=float)

    if corr.ndim == 0:
        return np.array([[1.0]])

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)

    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diag = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(diag, diag)

    np.fill_diagonal(corr_psd, 1.0)
    return corr_psd


def _sample_true_time_shift(
    rng: np.random.Generator,
    dx: str,
    time_shift_samples_by_dx: dict[str, np.ndarray],
) -> float:
    """Sample one subject-level DPMoSt time shift in months."""
    samples = time_shift_samples_by_dx.get(str(dx))

    if samples is not None:
        samples = np.asarray(samples, dtype=float)

        if len(samples) > 0:
            return float(rng.choice(samples))

    if dx not in DX_TIMESHIFT_PARAMS:
        # Use the fallback distribution for an unknown diagnosis.
        means = np.array([params[0] for params in DX_TIMESHIFT_PARAMS.values()])
        sds = np.array([params[1] for params in DX_TIMESHIFT_PARAMS.values()])
        mean = float(means.mean())
        sd = float(sds.mean())
    else:  # Set to a fallback normal distribution.
        mean, sd = DX_TIMESHIFT_PARAMS[dx]

    return float(rng.normal(mean, sd))


def _logistic_biomarker(
    stage: float,
    low: float,
    high: float,
    midpoint: float,
    slope: float,
    direction: int,
) -> float:
    """Generate a noiseless biomarker value from true disease stage (no timeshift)."""
    increasing = low + (high - low) / (1.0 + np.exp(-slope * (stage - midpoint)))

    if direction > 0:
        return float(increasing)

    return float(high + low - increasing)


def _fit_logistic_slope_midpoint(
    stages: pd.Series,
    values: pd.Series,
    low: float,
    high: float,
    direction: int,
) -> tuple[float, float]:
    """Learn logistic slope and midpoint for one biomarker.

    The real ADNI data do not contain true disease stage, so this uses the
    diagnosis-based pseudo-stage created in calibrate_from_adni().

    low/high are fixed to empirical biomarker quantiles. Only slope and midpoint
    are fitted.
    """
    fit_df = pd.DataFrame(
        {
            "stage": _safe_numeric(stages),
            "value": _safe_numeric(values),
        }
    ).dropna()

    if len(fit_df) < 10 or fit_df["stage"].nunique() < 3:
        return DEFAULT_LOGISTIC_SLOPE, DEFAULT_LOGISTIC_MIDPOINT

    x = fit_df["stage"].to_numpy(dtype=float)
    y = fit_df["value"].to_numpy(dtype=float)

    def curve(stage: np.ndarray, slope: float, midpoint: float) -> np.ndarray:
        return np.array(
            [
                _logistic_biomarker(
                    stage=float(s),
                    low=low,
                    high=high,
                    midpoint=midpoint,
                    slope=slope,
                    direction=direction,
                )
                for s in stage
            ],
            dtype=float,
        )

    midpoint_min = float(np.nanpercentile(x, 1) - 5.0)
    midpoint_max = float(np.nanpercentile(x, 99) + 5.0)

    # Ensure that the initial midpoint lies strictly inside the bounds, so don't
    # set a default value of DEFAULT_LOGISTIC_MIDPOINT = 0.0.
    # Instead, use the median of the data as the initial guess.
    initial_midpoint = float(
        np.clip(
            np.median(x),
            midpoint_min + 1e-6,
            midpoint_max - 1e-6,
        )
    )

    try:
        params, _ = curve_fit(
            curve,
            x,
            y,
            p0=[DEFAULT_LOGISTIC_SLOPE, initial_midpoint],
            bounds=([0.01, midpoint_min], [2.0, midpoint_max]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return DEFAULT_LOGISTIC_SLOPE, initial_midpoint

    slope, midpoint = params
    return float(slope), float(midpoint)


def _estimate_noise_sd(
    df: pd.DataFrame,
    subject_col: str,
    visit_month_col: str,
    biomarker: str,
) -> float:
    """Estimate simple measurement noise from within-subject longitudinal changes."""
    tmp = df[[subject_col, visit_month_col, biomarker]].copy()
    tmp[biomarker] = _safe_numeric(tmp[biomarker])
    tmp = tmp.dropna().sort_values([subject_col, visit_month_col])

    marginal_sd = tmp[biomarker].std(ddof=1)
    if pd.isna(marginal_sd) or marginal_sd <= 0:
        marginal_sd = 1.0

    # within-subject change in one biomarker between consecutive visits.
    diffs = tmp.groupby(subject_col)[biomarker].diff().dropna()
    if len(diffs) < 2:
        return float(0.10 * marginal_sd)

    # This includes true longitudinal biological change, so cap it.
    noise_sd = float(diffs.std(ddof=1) / np.sqrt(2.0))
    return float(np.clip(noise_sd, 1e-6, 0.50 * marginal_sd))


def _load_adni_learning_data(
    config_path: Path,
    adni_merge_csv: Path,
    freesurfer_csv: Path,
    explicit_measures: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Load raw ADNI files for simulator calibration.

    The simulator is meant to run before get_adni_data.py and output a new
    simulated ADNI folder with the same file names and columns.
    """
    # Step 1: Read config.json.
    config = load_json(config_path)
    measures = (
        list(explicit_measures) if explicit_measures else list(config["measures"])
    )

    # Step 2: Read the raw Freesurfer CSV (just needed for biomarkers' distributions).
    df_fs = pd.read_csv(freesurfer_csv, low_memory=False)

    # Step 3: Read ADNIMERGE for metadata distributions and output structure.
    df_merge = pd.read_csv(adni_merge_csv, low_memory=False)

    # Step 4: Validate required Freesurfer and ADNIMERGE columns.
    required_fs_cols = {
        FS_COL_SUBJECT,
        FS_COL_VISIT,
        FS_COL_AGE,
        FS_COL_SEX,
        FS_COL_DX,
    }
    missing_fs_cols = sorted(required_fs_cols - set(df_fs.columns))
    if missing_fs_cols:
        raise ValueError(f"{freesurfer_csv} is missing columns: {missing_fs_cols}")

    missing_measures = [col for col in measures if col not in df_fs.columns]
    if missing_measures:
        raise ValueError(
            "These config measures (biomarkers) are not columns in the raw Freesurfer CSV: "
            f"{missing_measures}"
        )

    required_merge_cols = {
        MERGE_COL_SUBJECT,
        MERGE_COL_VISIT,
        MERGE_COL_AGE,
        MERGE_COL_DX_BL,
    }
    missing_merge_cols = sorted(required_merge_cols - set(df_merge.columns))
    if missing_merge_cols:
        raise ValueError(f"{adni_merge_csv} is missing columns: {missing_merge_cols}")

    return df_fs, df_merge, measures, config


def calibrate_from_adni(
    config_path: Path,
    adni_merge_csv: Path,
    freesurfer_csv: Path,
    explicit_measures: tuple[str, ...],
    time_shift_samples_by_dx: dict[str, np.ndarray],
) -> tuple[Calibration, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Learn simple population-level distributions from raw ADNI files."""
    df_fs, df_merge, measures, config = _load_adni_learning_data(
        config_path=config_path,
        adni_merge_csv=adni_merge_csv,
        freesurfer_csv=freesurfer_csv,
        explicit_measures=explicit_measures,
    )

    df = df_fs.copy()

    # Step 1: Add parsed visit month, age, and map baseline diagnosis from ADNIMERGE.
    df["visit_month_sim"] = df[FS_COL_VISIT].map(_visit_to_month)
    df[FS_COL_AGE] = _safe_numeric(df[FS_COL_AGE])

    # Map the more specific baseline diagnosis from ADNIMERGE onto every
    # longitudinal Freesurfer row belonging to the same RID.
    dx_bl_by_subject = _get_baseline_dx_by_subject(df_merge)

    df["subject_key_sim"] = _normalize_subject_ids(df[FS_COL_SUBJECT])
    df["dx_sim"] = df["subject_key_sim"].map(dx_bl_by_subject)

    n_unmatched_rows = int(df["dx_sim"].isna().sum())
    if n_unmatched_rows:
        print(
            f"Dropping {n_unmatched_rows} Freesurfer rows whose RID has no "
            "usable ADNIMERGE DX_bl value corresponding to that subject."
        )

    # Only retain the specific baseline categories used by the DPMoSt priors:
    # CN, SMC, EMCI, LMCI, and AD.
    valid_dx = df["dx_sim"].isin(ALLOWED_BASELINE_DX)
    n_invalid_rows = int((~valid_dx).sum())

    if n_invalid_rows:
        invalid_labels = sorted(
            str(value) for value in df.loc[~valid_dx, "dx_sim"].dropna().unique()
        )
        print(
            f"Dropping {n_invalid_rows} rows outside ALLOWED_BASELINE_DX. "
            f"Observed labels: {invalid_labels}"
        )

    df = df.loc[valid_dx].copy()

    # Step 2: Define one baseline row per real subject.
    baseline = (
        df.sort_values([FS_COL_SUBJECT, "visit_month_sim"])
        .groupby(FS_COL_SUBJECT, as_index=False)
        .first()
        .dropna(subset=[FS_COL_AGE])
    )

    # Step 2b: Create an ADNI pseudo-stage for fitting biomarker curves.
    #
    # Real ADNI does not contain known latent stages. The diagnosis-specific
    # mean DPMoSt shift is used as the subject's approximate baseline position.
    # The empirical prior mean is preferred, with DX_TIMESHIFT_PARAMS used as
    # the fallback.
    stage_mean_years_by_dx = _get_stage_mean_years_by_dx(time_shift_samples_by_dx)

    def get_stage_mean(dx: Any) -> float:
        """Return mean baseline latent disease time in years."""
        dx = str(dx).strip().upper()

        if dx in stage_mean_years_by_dx:
            return stage_mean_years_by_dx[dx]

        # Defensive fallback (mean of allowed DX_bl means) for an unexpected diagnosis.
        # Normally, such rows have already been removed by ALLOWED_BASELINE_DX.
        allowed_means = [
            stage_mean_years_by_dx[diagnosis]
            for diagnosis in ALLOWED_BASELINE_DX
            if diagnosis in stage_mean_years_by_dx
        ]

        if not allowed_means:
            raise ValueError("No diagnosis-specific time-shift means available.")

        return float(np.mean(allowed_means))

    baseline_stage_mean_by_subject = baseline.set_index(FS_COL_SUBJECT)["dx_sim"].map(
        get_stage_mean
    )

    df["baseline_stage_mean_sim"] = df[FS_COL_SUBJECT].map(
        baseline_stage_mean_by_subject
    )

    # Logistic trajectory parameters are expressed in years.
    df["pseudo_stage_sim"] = (
        df["baseline_stage_mean_sim"] + df["visit_month_sim"].fillna(0.0) / 12.0
    )

    # Step 3: Learn cohort size and visit counts distribution.
    n_subjects = int(baseline[FS_COL_SUBJECT].nunique())

    visit_months = sorted(
        int(x) for x in df["visit_month_sim"].dropna().unique().tolist()
    )
    if not visit_months:
        visit_months = DEFAULT_VISIT_MONTHS

    visit_counts = df.groupby(FS_COL_SUBJECT).size()  # visits per subject.
    visit_count_probs = visit_counts.value_counts(normalize=True).sort_index().to_dict()

    # Step 4: Learn demographic and diagnosis distributions.
    sex_probs = (
        baseline[FS_COL_SEX].fillna("Unknown").value_counts(normalize=True).to_dict()
    )

    dx_probs = (
        baseline.loc[
            baseline["dx_sim"].isin(ALLOWED_BASELINE_DX),
            "dx_sim",
        ]
        .value_counts(normalize=True)
        .to_dict()
    )

    if not dx_probs:
        raise ValueError(
            "Could not estimate diagnosis probabilities from ADNIMERGE DX_bl."
        )

    age_mean = float(baseline[FS_COL_AGE].mean())
    age_sd = float(max(baseline[FS_COL_AGE].std(ddof=1), 1e-6))  # sample stddev (N-1)
    # percentile clipping to avoid extreme outliers
    age_min = float(baseline[FS_COL_AGE].quantile(0.01))
    age_max = float(baseline[FS_COL_AGE].quantile(0.99))

    if MERGE_COL_SITE in df_merge.columns:
        site_probs = (
            df_merge[MERGE_COL_SITE].astype(str).value_counts(normalize=True).to_dict()
        )
    else:
        site_probs = {"1": 1.0}

    # Step 5: Learn one simple logistic trajectory per configured biomarker.
    biomarker_params: dict[str, BiomarkerParams] = {}

    for biomarker in measures:
        values = _safe_numeric(df[biomarker])
        valid = values.dropna()

        if valid.empty:
            raise ValueError(f"Biomarker has no numeric values: {biomarker}")

        low = float(valid.quantile(0.01))
        high = float(valid.quantile(0.99))

        if np.isclose(low, high):
            sd = float(max(valid.std(ddof=1), 1e-6))
            low = float(valid.mean() - sd)
            high = float(valid.mean() + sd)

        cn_mean = _safe_numeric(df.loc[df["dx_sim"] == "CN", biomarker]).mean()
        ad_mean = _safe_numeric(df.loc[df["dx_sim"] == "AD", biomarker]).mean()

        # direction = -1 means biomarker decreases with disease stage.
        if pd.isna(cn_mean) or pd.isna(ad_mean):
            direction = -1
        else:
            direction = 1 if ad_mean > cn_mean else -1

        # Fit biomarker-specific logistic slope and midpoint from ADNI.
        #
        # This uses pseudo_stage_sim rather than true disease stage, because ADNI
        # does not contain ground-truth onset ages. The fitted values should be
        # interpreted as ADNI-calibrated baseline trajectory parameters, not as a
        # fully identified disease progression model.
        slope, midpoint = _fit_logistic_slope_midpoint(
            stages=df["pseudo_stage_sim"],
            values=df[biomarker],
            low=low,
            high=high,
            direction=direction,
        )

        print(
            f"{biomarker}: slope={slope:.4f}, "
            f"midpoint={midpoint:.4f} years, direction={direction}"
        )

        biomarker_params[biomarker] = BiomarkerParams(
            name=biomarker,
            low=low,
            high=high,
            slope=slope,
            midpoint=midpoint,
            direction=direction,
            noise_sd=_estimate_noise_sd(
                df=df,
                subject_col=FS_COL_SUBJECT,
                visit_month_col="visit_month_sim",
                biomarker=biomarker,
            ),
            marginal_mean=float(valid.mean()),
            marginal_sd=float(max(valid.std(ddof=1), 1e-6)),
        )

    # Step 6: Learn cross-biomarker correlation for correlated observation noise.
    biomarker_df = df[measures].apply(pd.to_numeric, errors="coerce")
    noise_corr = _nearest_psd_corr(biomarker_df.corr().to_numpy())

    calibration = Calibration(
        n_subjects=n_subjects,
        visit_months=visit_months,
        visit_count_probs={int(k): float(v) for k, v in visit_count_probs.items()},
        age_mean=age_mean,
        age_sd=age_sd,
        age_min=age_min,
        age_max=age_max,
        sex_probs={str(k): float(v) for k, v in sex_probs.items()},
        dx_probs={str(k): float(v) for k, v in dx_probs.items()},
        site_probs={str(k): float(v) for k, v in site_probs.items()},
        biomarkers=biomarker_params,
        noise_corr=noise_corr.tolist(),
    )

    return calibration, df_fs, df_merge, measures, config


def _sample_visit_months(
    rng: np.random.Generator,
    calibration: Calibration,
) -> list[int]:
    """Sample number of visits, then use earliest observed ADNI-like visits."""
    counts = np.array(list(calibration.visit_count_probs.keys()), dtype=int)
    probs = np.array(list(calibration.visit_count_probs.values()), dtype=float)
    probs = probs / probs.sum()

    n_visits = int(rng.choice(counts, p=probs))
    n_visits = int(np.clip(n_visits, 1, len(calibration.visit_months)))

    return calibration.visit_months[:n_visits]


def simulate_raw_adni_files(
    calibration: Calibration,
    fs_columns: list[str],
    merge_columns: list[str],
    measures: list[str],
    n_subjects: int | None,
    rng_seed: int,
    time_shift_samples_by_dx: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate simulated raw Freesurfer and ADNIMERGE-like files."""
    rng = np.random.default_rng(rng_seed)

    if n_subjects is None:
        n_subjects = calibration.n_subjects

    biomarker_names = list(calibration.biomarkers)

    # Step 1: Build correlated noise covariance for configured biomarkers.
    noise_sds = np.array(
        [calibration.biomarkers[name].noise_sd for name in biomarker_names],
        dtype=float,
    )
    noise_corr = np.array(calibration.noise_corr, dtype=float)
    noise_cov = noise_corr * np.outer(noise_sds, noise_sds)

    fs_rows = []
    merge_rows = []

    for subject_idx in range(n_subjects):
        # Step 2: Create a new virtual RID.
        rid = subject_idx + 1

        # Step 3: Sample baseline-level variables.
        dx = _sample_categorical(rng, calibration.dx_probs)
        sex = _sample_categorical(rng, calibration.sex_probs)
        site = _sample_categorical(rng, calibration.site_probs)

        baseline_age = float(
            np.clip(
                rng.normal(calibration.age_mean, calibration.age_sd),
                calibration.age_min,
                calibration.age_max,
            )
        )

        true_time_shift = _sample_true_time_shift(
            rng=rng,
            dx=str(dx),
            time_shift_samples_by_dx=time_shift_samples_by_dx,
        )

        # Derived age at latent disease-time zero.
        true_onset_age = baseline_age - true_time_shift / 12.0

        visit_months = _sample_visit_months(rng, calibration)

        for visit_month in visit_months:
            fs_visit = _month_to_visit(visit_month)
            merge_visit = _month_to_merge_visit(visit_month)

            # Step 4: Compute hidden disease stage for this visit.
            # age_at_visit = baseline_age + visit_month / 12.0
            true_stage = (true_time_shift + visit_month) / 12.0

            # Step 5: Generate noiseless configured biomarkers.
            noiseless = np.array(
                [
                    _logistic_biomarker(
                        stage=true_stage,
                        low=calibration.biomarkers[name].low,
                        high=calibration.biomarkers[name].high,
                        midpoint=calibration.biomarkers[name].midpoint,
                        slope=calibration.biomarkers[name].slope,
                        direction=calibration.biomarkers[name].direction,
                    )
                    for name in biomarker_names
                ],
                dtype=float,
            )

            # Step 6: Add correlated ADNI-calibrated measurement noise.
            noise = rng.multivariate_normal(
                mean=np.zeros(len(biomarker_names)),
                cov=noise_cov,
            )
            observed = noiseless + noise

            # Step 7: Create one Freesurfer-format row.
            # All original columns are preserved. Only required metadata and
            # configured measures are populated in this baseline simulator.
            fs_row = {col: np.nan for col in fs_columns}
            fs_row[FS_COL_SUBJECT] = rid
            fs_row[FS_COL_VISIT] = fs_visit
            fs_row[FS_COL_AGE] = baseline_age
            fs_row[FS_COL_SEX] = sex
            fs_row[FS_COL_DX] = dx

            for name, value in zip(biomarker_names, observed):
                fs_row[name] = value

            fs_row[COL_TRUE_ONSET_AGE] = true_onset_age
            fs_row[COL_TRUE_DISEASE_STAGE] = true_stage
            fs_row[COL_TRUE_TIME_SHIFT] = true_time_shift

            fs_rows.append(fs_row)

            # Step 8: Create one ADNIMERGE-format row.
            # This preserves the raw ADNIMERGE columns expected by get_adni_data.py.
            merge_row = {col: np.nan for col in merge_columns}
            merge_row[MERGE_COL_SUBJECT] = rid
            merge_row[MERGE_COL_VISIT] = merge_visit
            merge_row[MERGE_COL_AGE] = baseline_age

            if MERGE_COL_SITE in merge_columns:
                merge_row[MERGE_COL_SITE] = site
            if MERGE_COL_DX in merge_columns:
                merge_row[MERGE_COL_DX] = dx
            if MERGE_COL_DX_BL in merge_columns:
                merge_row[MERGE_COL_DX_BL] = dx

            merge_rows.append(merge_row)

    df_fs_sim = pd.DataFrame(fs_rows)

    # Keep original Freesurfer columns first, then append ground-truth columns.
    truth_cols = [
        COL_TRUE_ONSET_AGE,
        COL_TRUE_DISEASE_STAGE,
        COL_TRUE_TIME_SHIFT,
    ]
    df_fs_sim = df_fs_sim[fs_columns + truth_cols]

    df_merge_sim = pd.DataFrame(merge_rows)
    df_merge_sim = df_merge_sim[merge_columns]

    return df_fs_sim, df_merge_sim


def _json_ready(value: Any) -> Any:
    """Convert dataclass/NumPy/Path objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _calibration_to_json(calibration: Calibration) -> dict[str, Any]:
    data = asdict(calibration)
    data["biomarkers"] = {
        name: asdict(params) for name, params in calibration.biomarkers.items()
    }
    return _json_ready(data)


def _build_mirrored_params(
    calibration: Calibration,
    config: dict[str, Any],
    df_fs_sim: pd.DataFrame,
) -> dict[str, Any]:
    """Build a simulate_data.py-style ``params`` entry in model space.

    Time is scaled by ``config['max_time']`` (months -> model time units),
    matching the transform applied by get_adni_data.py. Per-biomarker arrays
    follow sorted-measure order; time shifts and acceleration factors are
    keyed by normalized RID. vertical_shifts and scaling_factors are fixed to
    the model's parametrization (0 and 1). sigmas are the generative noise
    standard deviations rescaled to the min-max-normalized observation space.
    """
    t_unit = config.get("max_time") or 1.0

    time_shifts_months = (
        df_fs_sim.groupby(FS_COL_SUBJECT)[COL_TRUE_TIME_SHIFT].first().astype(float)
    )

    params: dict[str, Any] = {
        "time_shifts": {
            str(rid): float(shift_months / t_unit)
            for rid, shift_months in time_shifts_months.items()
        },
        "acceleration_factors": {str(rid): 1.0 for rid in time_shifts_months.index},
        "k_values": [],
        "x0_values": [],
        "vertical_shifts": [],
        "scaling_factors": [],
        "sigmas": [],
    }

    min_max_by_measure = config.get("min_max_by_measure", {})

    for measure in sorted(calibration.biomarkers):
        biomarker = calibration.biomarkers[measure]

        if measure not in min_max_by_measure:
            raise ValueError(
                f"config['min_max_by_measure'] is missing an entry for "
                f"{measure!r}, which is required to express sigmas in model space."
            )

        value_min, value_max = min_max_by_measure[measure]
        value_range = abs(value_max - value_min)

        params["k_values"].append(biomarker.slope * t_unit / 12.0)
        params["x0_values"].append(biomarker.midpoint * 12.0 / t_unit)
        params["vertical_shifts"].append(0.0)
        params["scaling_factors"].append(1.0)
        params["sigmas"].append(biomarker.noise_sd / value_range)

    return params


def simulate_data_adni_timeshift(
    tag: str,
    dpath_data: Path,
    config: Path,
    adni_merge_csv: Path,
    measures_csv: Path,
    time_shift_priors: Path | None,
    measure: tuple[str, ...],
    n_subjects: int | None,
    rng_seed: int,
):
    """Generate a simulated ADNI folder that can be passed to get_adni_data.py."""
    freesurfer_csv = measures_csv

    time_shift_samples_by_dx, _time_shift_prior_metadata = _load_time_shift_priors(
        time_shift_priors
    )

    dpath_out = dpath_data / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    calibration, df_fs, df_merge, measures, config_data = calibrate_from_adni(
        config_path=config,
        adni_merge_csv=adni_merge_csv,
        freesurfer_csv=freesurfer_csv,
        explicit_measures=measure,
        time_shift_samples_by_dx=time_shift_samples_by_dx,
    )

    df_fs_sim, df_merge_sim = simulate_raw_adni_files(
        calibration=calibration,
        fs_columns=list(df_fs.columns),
        merge_columns=list(df_merge.columns),
        measures=measures,
        n_subjects=n_subjects,
        rng_seed=rng_seed,
        time_shift_samples_by_dx=time_shift_samples_by_dx,
    )

    # Write simulated raw ADNI-like files.
    fpath_fs_out = dpath_out / Path(freesurfer_csv).name
    fpath_merge_out = dpath_out / Path(adni_merge_csv).name
    fpath_config_out = dpath_out / Path(config).name
    fpath_metadata_out = dpath_out / "simulation_metadata.json"

    df_fs_sim.to_csv(fpath_fs_out, index=False)
    df_merge_sim.to_csv(fpath_merge_out, index=False)

    # Keep config unchanged so get_adni_data.py sees the same measure list.
    save_json(fpath_config_out, config_data)

    mirrored_params = _build_mirrored_params(
        calibration=calibration,
        config=config_data,
        df_fs_sim=df_fs_sim,
    )

    metadata = {
        "description": (
            "Simulated raw ADNI-like folder generated before get_adni_data.py. "
            "Only config['measures'] biomarkers are simulated; other raw columns "
            "are preserved but mostly left missing."
        ),
        "settings": {
            "tag": tag,
            "rng_seed": rng_seed,
            "n_subjects": n_subjects,
            "source_config": str(config),
            "source_adni_merge_csv": str(adni_merge_csv),
            "source_freesurfer_csv": str(freesurfer_csv),
            "output_dir": str(dpath_out),
        },
        "measures": measures,
        "ground_truth_columns_in_freesurfer_csv": [
            COL_TRUE_ONSET_AGE,
            COL_TRUE_DISEASE_STAGE,
            COL_TRUE_TIME_SHIFT,
        ],
        "params": mirrored_params,
        "calibration": _calibration_to_json(calibration),
    }
    save_json(fpath_metadata_out, metadata)

    print(f"Saved simulated Freesurfer CSV: {fpath_fs_out}")
    print(f"Saved simulated ADNIMERGE CSV:   {fpath_merge_out}")
    print(f"Saved copied config JSON:        {fpath_config_out}")
    print(f"Saved simulation metadata:       {fpath_metadata_out}")
    print()
    print("Next, run get_adni_data.py using these simulated files.")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, default=DEFAULT_TAG, show_default=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_CONFIG_FILE",
)
@click.option(
    "--adni-merge-csv",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_MERGE_FILE",
)
@click.option(
    "--measures-csv",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_IDP_FILE",
)
@click.option(
    "--time-shift-priors",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=DEFAULT_TIME_SHIFT_PRIORS_JSON,
    show_default=True,
    help=(
        "Diagnosis-specific empirical DPMoSt time-shift priors. "
        "Falls back to DX_TIMESHIFT_PARAMS when unavailable."
    ),
)
@click.option(
    "--measure",
    type=str,
    multiple=True,
    default=(),
    help="Optional override for config['measures']. Normally leave empty.",
)
@click.option(
    "--n-subjects",
    type=click.IntRange(min=1),
    default=None,
    help="Synthetic subject count. Defaults to learned ADNI subject count.",
)
@click.option("--rng-seed", type=int, default=DEFAULT_RNG_SEED, show_default=True)
def main(**params):
    simulate_data_adni_timeshift(**params)


if __name__ == "__main__":
    main()

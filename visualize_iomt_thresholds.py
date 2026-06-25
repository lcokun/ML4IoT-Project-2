from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "iomt-alert" / "patients_data_with_alerts.xlsx"
OUTPUT_DIR = Path(__file__).parent / "output"

HR_COL = "Heart Rate (bpm)"
SPO2_COL = "SpO2 Level (%)"
SYSTOLIC_COL = "Systolic Blood Pressure (mmHg)"
DIASTOLIC_COL = "Diastolic Blood Pressure (mmHg)"
TEMP_COL = "Body Temperature (°C)"
FALL_COL = "Fall Detection"

VITAL_COLUMNS = [HR_COL, SPO2_COL, SYSTOLIC_COL, DIASTOLIC_COL, TEMP_COL]

# (low, high) bound mirrored from assign_triage() in Preprocess_MLIoT_Dataset.ipynb
EMERGENCY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    HR_COL: (40, 130),
    SPO2_COL: (90, None),
    SYSTOLIC_COL: (90, 180),
    DIASTOLIC_COL: (None, 120),
    TEMP_COL: (35, 39.5),
}
URGENT_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    HR_COL: (50, 110),
    SPO2_COL: (94, None),
    SYSTOLIC_COL: (100, 140),
    DIASTOLIC_COL: (None, 90),
    TEMP_COL: (36, 38),
}


def load_iomt_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def assign_triage(row: pd.Series) -> str:
    """Mirrors Preprocess_MLIoT_Dataset.ipynb cell 13 — kept in sync manually."""
    heart_rate = row[HR_COL]
    spo2 = row[SPO2_COL]
    systolic = row[SYSTOLIC_COL]
    diastolic = row[DIASTOLIC_COL]
    temperature = row[TEMP_COL]
    fall = str(row[FALL_COL]).strip().lower() == "yes"

    if (
        spo2 < 90
        or heart_rate < 40
        or heart_rate >= 130
        or systolic < 90
        or systolic >= 180
        or diastolic >= 120
        or temperature < 35
        or temperature >= 39.5
    ):
        return "Emergency"
    if (
        spo2 < 94
        or heart_rate < 50
        or heart_rate >= 110
        or systolic < 100
        or systolic >= 140
        or diastolic >= 90
        or temperature < 36
        or temperature >= 38
        or fall
    ):
        return "Urgent"
    return "Non-Urgent"


def condition_trigger_counts(
    df: pd.DataFrame, bounds: dict[str, tuple[float | None, float | None]]
) -> pd.Series:
    """Count how many rows breach each vital's low/high bound, individually."""
    counts = {}
    for col, (low, high) in bounds.items():
        if low is not None:
            counts[f"{col} < {low}"] = int((df[col] < low).sum())
        if high is not None:
            counts[f"{col} >= {high}"] = int((df[col] >= high).sum())
    return pd.Series(counts)


def plot_vital_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Histogram per vital, with Emergency (red) and Urgent (orange) zones shaded."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, VITAL_COLUMNS):
        data = df[col]
        ax.hist(data, bins=40, color="steelblue", edgecolor="white")

        em_low, em_high = EMERGENCY_BOUNDS[col]
        ur_low, ur_high = URGENT_BOUNDS[col]
        x_min, x_max = data.min(), data.max()

        if ur_low is not None:
            ax.axvspan(x_min, ur_low, color="orange", alpha=0.15)
        if ur_high is not None:
            ax.axvspan(ur_high, x_max, color="orange", alpha=0.15)
        if em_low is not None:
            ax.axvspan(x_min, em_low, color="red", alpha=0.15)
        if em_high is not None:
            ax.axvspan(em_high, x_max, color="red", alpha=0.15)

        ax.set_title(col, fontsize=10)
        ax.set_xlim(x_min, x_max)

    axes[-1].axis("off")
    fig.suptitle(
        "IoMT vital distributions vs assign_triage() thresholds\n"
        "(red = Emergency zone, orange = Urgent zone — note how often they're empty)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trigger_rates(
    emergency_counts: pd.Series,
    urgent_counts: pd.Series,
    n_total: int,
    n_non_emergency: int,
    output_path: Path,
) -> None:
    """Bar chart of how often each individual sub-condition actually fires."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    em_pct = (emergency_counts / n_total * 100).sort_values()
    axes[0].barh(em_pct.index, em_pct.values, color="firebrick")
    axes[0].set_title(f"Emergency sub-conditions\n(% of all {n_total} rows)")
    axes[0].set_xlabel("% of rows triggered")
    for i, v in enumerate(em_pct.values):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center")

    ur_pct = (urgent_counts / n_non_emergency * 100).sort_values()
    axes[1].barh(ur_pct.index, ur_pct.values, color="darkorange")
    axes[1].set_title(f"Urgent sub-conditions\n(% of {n_non_emergency} non-Emergency rows)")
    axes[1].set_xlabel("% of rows triggered")
    for i, v in enumerate(ur_pct.values):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center")

    fig.suptitle("Which assign_triage() conditions actually fire in this dataset", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_triage_distribution(triage: pd.Series, output_path: Path) -> None:
    counts = triage.value_counts()
    colors = {"Emergency": "firebrick", "Urgent": "darkorange", "Non-Urgent": "seagreen"}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index, counts.values, color=[colors[label] for label in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v}\n({v / counts.sum() * 100:.1f}%)", ha="center")
    ax.set_title("Final Triage label distribution")
    ax.set_ylabel("Row count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_iomt_data(DATA_PATH)
    df["Triage"] = df.apply(assign_triage, axis=1)

    non_emergency = df[df["Triage"] != "Emergency"]
    emergency_counts = condition_trigger_counts(df, EMERGENCY_BOUNDS)
    urgent_counts = condition_trigger_counts(non_emergency, URGENT_BOUNDS)
    urgent_counts[f"{FALL_COL} == Yes"] = int(
        (non_emergency[FALL_COL].str.strip().str.lower() == "yes").sum()
    )

    plot_vital_distributions(df, OUTPUT_DIR / "iomt_vital_distributions.png")
    plot_trigger_rates(
        emergency_counts,
        urgent_counts,
        len(df),
        len(non_emergency),
        OUTPUT_DIR / "iomt_condition_trigger_rates.png",
    )
    plot_triage_distribution(df["Triage"], OUTPUT_DIR / "iomt_triage_distribution.png")

    print(f"Saved 3 figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

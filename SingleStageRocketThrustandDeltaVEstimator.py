
"""
Rocket Thrust & Δv Estimator (single-stage)

What it does
------------
1) Direct compute from masses:
   - Give Isp and either (m0,mf) OR (dry,prop).
   - Prints Δv, mass ratio, thrust (if mdot given), burn time.

2) Target Δv solver:
   - Give Isp, dry mass, and target Δv -> returns required prop mass.

3) Sweep/plot:
   - Δv vs propellant mass fraction for a fixed dry mass and Isp.

Sane units:
- Mass in kg, Isp in seconds, mdot in kg/s, Δv in m/s (prints km/s too).

Examples
--------
# Direct compute (Isp=300 s, dry=20 t, prop=30 t, mdot=250 kg/s)
python rocket_estimator.py --isp 300 --dry 20000 --prop 30000 --mdot 250 --save csv plots

# From m0/mf instead
python rocket_estimator.py --isp 300 --m0 50000 --mf 20000

# Solve required propellant for Δv target (6.0 km/s)
python rocket_estimator.py --isp 320 --dry 15000 --target-dv 6000

# Sweep Δv vs prop fraction (5–95%) for a 2 t dry mass, Isp=310 s
python rocket_estimator.py --isp 310 --dry 2000 --sweep --sweep-min 0.05 --sweep-max 0.95 --sweep-step 0.05 --save plots csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665  # m/s^2 (standard gravity)


# Core relations
def delta_v(isp_s: float, m0: float, mf: float) -> float:
    """Tsiolkovsky: Δv [m/s] for Isp [s], initial mass m0, final mass mf."""
    if m0 <= 0 or mf <= 0 or m0 <= mf:
        raise ValueError("Require m0 > mf > 0.")
    return isp_s * G0 * np.log(m0 / mf)


def mass_ratio_from_dv(dv_ms: float, isp_s: float) -> float:
    """Return MR = m0/mf required to achieve dv (m/s) at Isp (s)."""
    return float(np.exp(dv_ms / (isp_s * G0)))


def prop_required_for_dv(dry_mass: float, dv_ms: float, isp_s: float) -> tuple[float, float, float]:
    """
    For a given dry mass and target Δv, compute:
    (prop_mass, m0, mf).
    """
    if dry_mass <= 0:
        raise ValueError("dry_mass must be > 0")
    MR = mass_ratio_from_dv(dv_ms, isp_s)
    mf = dry_mass
    m0 = MR * mf
    prop = m0 - mf
    return prop, m0, mf


def thrust_newton(isp_s: float, mdot: float) -> float:
    """Thrust [N] from Isp [s] and mass flow rate mdot [kg/s] (vacuum thrust approximation)."""
    if mdot <= 0:
        raise ValueError("mdot must be > 0")
    return mdot * isp_s * G0


# IO helpers
def save_csv_rows(path: Path, header: list[str], rows: list[list[float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"✓ CSV saved -> {path}")


def ensure_positive(name: str, val: float | None):
    if val is None:
        return
    if val <= 0:
        raise SystemExit(f"Error: {name} must be > 0 (got {val}).")


# CLI 
def parse_args():
    p = argparse.ArgumentParser(description="Rocket thrust & Δv estimator (single-stage)")
    p.add_argument("--isp", type=float, required=True, help="specific impulse [s]")

    # Mass specification (choose one way)
    p.add_argument("--m0", type=float, help="initial mass [kg]")
    p.add_argument("--mf", type=float, help="final mass [kg] (dry + payload)")
    p.add_argument("--dry", type=float, help="dry mass [kg] (structure + payload)")
    p.add_argument("--prop", type=float, help="propellant mass [kg]")

    # Thrust inputs
    p.add_argument("--mdot", type=float, help="mass flow rate [kg/s] to compute thrust")
    p.add_argument("--burn", action="store_true", help="also compute burn time if prop mass known")

    # Target Δv solver
    p.add_argument("--target-dv", type=float, help="target Δv to achieve [m/s] (solves required prop for given dry)")

    # Sweep options
    p.add_argument("--sweep", action="store_true", help="plot Δv vs propellant mass fraction for fixed dry & Isp")
    p.add_argument("--sweep-min", type=float, default=0.05, help="min prop fraction (0–1)")
    p.add_argument("--sweep-max", type=float, default=0.95, help="max prop fraction (0–1)")
    p.add_argument("--sweep-step", type=float, default=0.05, help="step in prop fraction")

    # Save outputs
    p.add_argument("--save", nargs="*", choices=["csv", "plots"], help="save outputs")
    p.add_argument("--out", default="outputs", help="output directory")

    return p.parse_args()


# Plotting 
def plot_sweep(dry: float, isp: float, pf_array: np.ndarray, outdir: Path | None):
    m0 = dry / (1 - pf_array)
    mf = np.full_like(m0, dry)
    dv = delta_v(isp, m0, mf)

    # Δv vs prop fraction
    plt.figure()
    plt.plot(pf_array * 100.0, dv / 1000.0)
    plt.xlabel("Propellant mass fraction (%)")
    plt.ylabel("Δv (km/s)")
    plt.title(f"Δv vs propellant fraction (dry={dry:.0f} kg, Isp={isp:.0f} s)")
    plt.grid(True)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "dv_vs_prop_fraction.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"✓ Plot saved -> {path}")
    else:
        plt.show()

    return dv, m0, mf


# Main 
def main():
    args = parse_args()
    ensure_positive("isp", args.isp)
    if args.m0: ensure_positive("m0", args.m0)
    if args.mf: ensure_positive("mf", args.mf)
    if args.dry: ensure_positive("dry", args.dry)
    if args.prop: ensure_positive("prop", args.prop)
    if args.mdot: ensure_positive("mdot", args.mdot)

    outdir = Path(args.out)
    rows = []

    # 1) Target Δv solver (needs dry & target-dv)
    if args.target-dv is not None:
        if not args.dry:
            raise SystemExit("For --target-dv you must also provide --dry.")
        prop, m0, mf = prop_required_for_dv(args.dry, args.target-dv, args.isp)
        print("\n=== Target Δv Solver ===")
        print(f"Isp         : {args.isp:.1f} s")
        print(f"Target Δv   : {args.target-dv:.0f} m/s ({args.target-dv/1000:.2f} km/s)")
        print(f"Dry mass    : {args.dry:.1f} kg")
        print(f"Prop req'd  : {prop:.1f} kg")
        print(f"m0 (lift-off): {m0:.1f} kg   |   mf (dry): {mf:.1f} kg")
        rows.append(["solver", args.isp, args.target-dv, args.dry, prop, m0, mf, "", "", ""])

        if args.save and "csv" in args.save:
            save_csv_rows(outdir / "target_dv_solution.csv",
                          ["mode","Isp_s","target_dv_m_s","dry_kg","prop_kg","m0_kg","mf_kg","mdot_kg_s","thrust_N","burn_s"],
                          rows)

    # 2) Direct compute (masses known)
    #    Accept (m0,mf) or (dry,prop)
    if (args.m0 and args.mf) or (args.dry and args.prop):
        if args.m0 and args.mf:
            m0, mf = args.m0, args.mf
            dry = mf
            prop = m0 - mf
        else:
            dry = args.dry
            prop = args.prop
            m0 = dry + prop
            mf = dry

        dv = delta_v(args.isp, m0, mf)
        MR = m0 / mf
        print("\n=== Direct Performance ===")
        print(f"Isp      : {args.isp:.1f} s")
        print(f"m0 / mf  : {MR:.4f}")
        print(f"Δv       : {dv:.0f} m/s ({dv/1000:.2f} km/s)")
        print(f"m0, mf   : {m0:.1f} kg, {mf:.1f} kg   |   prop: {prop:.1f} kg")

        thrust = None
        burn_s = None
        if args.mdot:
            thrust = thrust_newton(args.isp, args.mdot)
            print(f"Thrust   : {thrust/1000:.2f} kN  (mdot={args.mdot:.2f} kg/s)")
            if args.burn:
                burn_s = prop / args.mdot
                print(f"Burn time: {burn_s:.1f} s")

        rows2 = [["direct", args.isp, dv, dry, prop, m0, mf, args.mdot or "", thrust or "", burn_s or ""]]
        if args.save and "csv" in args.save:
            save_csv_rows(outdir / "direct_results.csv",
                          ["mode","Isp_s","dv_m_s","dry_kg","prop_kg","m0_kg","mf_kg","mdot_kg_s","thrust_N","burn_s"],
                          rows2)

    # 3) Sweep Δv vs prop fraction (requires dry & Isp)
    if args.sweep:
        if not args.dry:
            raise SystemExit("For --sweep you must provide --dry.")
        pf = np.arange(args.sweep-min, args.sweep-max + 1e-9, args.sweep-step)  # type: ignore
        # (arg names with hyphens become attributes with hyphens replaced by underscores)
        pf = np.arange(args.sweep_min, args.sweep_max + 1e-12, args.sweep_step)
        pf = np.clip(pf, 1e-6, 0.999999)
        dv, m0_arr, mf_arr = plot_sweep(args.dry, args.isp, pf, outdir if (args.save and "plots" in args.save) else None)
        if args.save and "csv" in args.save:
            rows = [["sweep", args.isp, dvi, args.dry, (m0i - args.dry), m0i, args.dry, "", "", ""]
                    for dvi, m0i in zip(dv, m0_arr)]
            save_csv_rows(outdir / "sweep_results.csv",
                          ["mode","Isp_s","dv_m_s","dry_kg","prop_kg","m0_kg","mf_kg","mdot_kg_s","thrust_N","burn_s"],
                          rows)

    if not (args.target-dv or (args.m0 and args.mf) or (args.dry and args.prop) or args.sweep):
        print("\nNothing to do. Provide either:")
        print("  --m0 & --mf   OR   --dry & --prop   OR   --target-dv (with --dry)   OR   --sweep (with --dry).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single-Stage Rocket Thrust and Δv Estimator (stable)

Modes
-----
1) Direct compute (known masses)
   - Use either --m0 & --mf  OR  --dry & --prop
   - Optional: --mdot (thrust), --burn (burn time)

2) Target Δv solver
   - Use --target-dv (m/s) with --dry (kg) to get required propellant

3) Sweep (Δv vs propellant mass fraction)
   - Use --sweep with --sweep-min/--sweep-max/--sweep-step (fractions 0–1)
   - Requires --dry and --isp
   - Saves PNG/CSV when --save is used

Outputs saved under:  outputs/
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

G0 = 9.80665  # m/s^2


# ---------- physics ----------
def delta_v(isp_s: float, m0: float | np.ndarray, mf: float | np.ndarray) -> float | np.ndarray:
    if np.any(np.asarray(m0) <= 0) or np.any(np.asarray(mf) <= 0):
        raise ValueError("m0 and mf must be > 0.")
    if np.any(np.asarray(m0) <= np.asarray(mf)):
        raise ValueError("Require m0 > mf.")
    return isp_s * G0 * np.log(np.asarray(m0) / np.asarray(mf))


def thrust_newton(isp_s: float, mdot: float) -> float:
    if mdot <= 0:
        raise ValueError("mdot must be > 0.")
    return mdot * isp_s * G0


def prop_required_for_dv(dry_mass: float, dv_ms: float, isp_s: float) -> tuple[float, float, float]:
    if dry_mass <= 0:
        raise ValueError("dry_mass must be > 0.")
    MR = float(np.exp(dv_ms / (isp_s * G0)))
    mf = dry_mass
    m0 = MR * mf
    prop = m0 - mf
    return prop, m0, mf


# ---------- io helpers ----------
def save_csv(path: Path, header: list[str], rows: list[list[object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"✓ CSV saved -> {path.resolve()}")


def ensure_pos(name: str, val: float | None):
    if val is None:
        return
    if val <= 0:
        raise SystemExit(f"{name} must be > 0 (got {val}).")


# ---------- cli ----------
def parse_args():
    p = argparse.ArgumentParser(description="Single-stage rocket thrust & Δv estimator")
    p.add_argument("--isp", type=float, required=True, help="specific impulse [s]")

    # mass specs
    p.add_argument("--m0", type=float, help="initial mass [kg]")
    p.add_argument("--mf", type=float, help="final mass [kg] (dry + payload)")
    p.add_argument("--dry", type=float, help="dry mass [kg] (structure + payload)")
    p.add_argument("--prop", type=float, help="propellant mass [kg]")

    # thrust/burn
    p.add_argument("--mdot", type=float, help="mass flow rate [kg/s]")
    p.add_argument("--burn", action="store_true", help="compute burn time if prop known")

    # target dv
    p.add_argument("--target-dv", dest="target_dv", type=float, help="target Δv to achieve [m/s]")

    # sweep settings (propellant fraction)
    p.add_argument("--sweep", action="store_true", help="Δv vs propellant mass fraction")
    p.add_argument("--sweep-min", type=float, default=0.05, help="min prop fraction (0–1)")
    p.add_argument("--sweep-max", type=float, default=0.95, help="max prop fraction (0–1)")
    p.add_argument("--sweep-step", type=float, default=0.05, help="step in prop fraction (0–1)")

    # saving
    p.add_argument("--save", nargs="*", choices=["csv", "plots"], help="save outputs")
    p.add_argument("--out", default="outputs", help="output directory")
    return p.parse_args()


# ---------- main ----------
def main():
    args = parse_args()
    ensure_pos("isp", args.isp)
    for k in ("m0", "mf", "dry", "prop", "mdot"):
        ensure_pos(k, getattr(args, k))

    outdir = Path(args.out)

    # --- target dv solver ---
    if args.target_dv is not None:
        if args.dry is None:
            raise SystemExit("For --target-dv you must also provide --dry.")
        prop, m0, mf = prop_required_for_dv(args.dry, args.target_dv, args.isp)
        print("\n=== Target Δv Solver ===")
        print(f"Isp         : {args.isp:.1f} s")
        print(f"Target Δv   : {args.target_dv:.0f} m/s ({args.target_dv/1000:.2f} km/s)")
        print(f"Dry mass    : {args.dry:.1f} kg")
        print(f"Prop req'd  : {prop:.1f} kg")
        print(f"m0 (lift-off): {m0:.1f} kg   |   mf (dry): {mf:.1f} kg")
        if args.save and "csv" in args.save:
            save_csv(outdir / "target_dv_solution.csv",
                     ["Isp_s","target_dv_m_s","dry_kg","prop_kg","m0_kg","mf_kg"],
                     [[args.isp, args.target_dv, args.dry, prop, m0, mf]])

    # --- direct compute ---
    did_direct = False
    if (args.m0 is not None and args.mf is not None) or (args.dry is not None and args.prop is not None):
        if args.m0 is not None:
            m0, mf = args.m0, args.mf
            dry = mf
            prop = m0 - mf
        else:
            dry = args.dry
            prop = args.prop
            m0 = dry + prop
            mf = dry

        dv = float(delta_v(args.isp, m0, mf))
        MR = m0 / mf
        print("\n=== Direct Performance ===")
        print(f"Isp      : {args.isp:.1f} s")
        print(f"m0 / mf  : {MR:.4f}")
        print(f"Δv       : {dv:.0f} m/s ({dv/1000:.2f} km/s)")
        print(f"masses   : m0={m0:.1f} kg, mf={mf:.1f} kg, prop={prop:.1f} kg")

        thrust = burn_s = ""
        if args.mdot is not None:
            F = thrust_newton(args.isp, args.mdot)
            thrust = f"{F:.3f}"
            print(f"Thrust   : {F/1000:.2f} kN  (mdot={args.mdot:.2f} kg/s)")
            if args.burn:
                burn = prop / args.mdot
                burn_s = f"{burn:.3f}"
                print(f"Burn time: {burn:.1f} s")

        if args.save and "csv" in args.save:
            save_csv(outdir / "direct_results.csv",
                     ["Isp_s","dv_m_s","dry_kg","prop_kg","m0_kg","mf_kg","mdot_kg_s","thrust_N","burn_s"],
                     [[args.isp, dv, dry, prop, m0, mf, args.mdot or "", thrust, burn_s]])
        did_direct = True

    # --- sweep (Δv vs propellant fraction) ---
    if args.sweep:
        if args.dry is None:
            raise SystemExit("For --sweep you must provide --dry.")
        pf_min = float(np.clip(args.sweep_min, 1e-6, 0.999999))
        pf_max = float(np.clip(args.sweep_max, pf_min + 1e-6, 0.999999))
        step   = float(np.clip(args.sweep_step, 1e-6, 1.0))
        pf = np.arange(pf_min, pf_max + 1e-12, step, dtype=float)
        pf = np.clip(pf, 1e-6, 0.999999)

        m0_arr = args.dry / (1.0 - pf)
        mf_arr = np.full_like(m0_arr, args.dry)
        dv_arr = delta_v(args.isp, m0_arr, mf_arr)

        if args.save and "csv" in args.save:
            rows = [["sweep", float(pfi), float(m0i), float(args.dry), float(m0i-args.dry), float(dvi)]
                    for pfi, m0i, dvi in zip(pf, m0_arr, dv_arr)]
            save_csv(outdir / "sweep_results.csv",
                     ["mode","prop_frac","m0_kg","mf_kg","prop_kg","dv_m_s"],
                     rows)

        if args.save and "plots" in args.save:
            import matplotlib.pyplot as plt
            outdir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.plot(pf * 100.0, dv_arr / 1000.0, color="blue", linewidth=2, label="Δv curve")
            plt.xlabel("Propellant mass fraction (%)")
            plt.ylabel("Δv (km/s)")
            plt.title(f"Δv vs Propellant Fraction\n(dry={args.dry:.0f} kg, Isp={args.isp:.0f} s)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()

            path = outdir / "dv_vs_prop_fraction.png"
            plt.savefig(path, dpi=300)
            plt.close()

            print(f"✓ Plot saved -> {path.resolve()}")

        else:
            # show numeric table if not saving a plot
            print("\n=== Sweep (Δv vs prop fraction) ===")
            print("  prop_frac    m0[kg]    Δv[km/s]")
            for pfi, m0i, dvi in zip(pf, m0_arr, dv_arr):
                print(f"   {pfi:8.3f}  {m0i:9.1f}   {dvi/1000:7.3f}")

    if not (args.target_dv is not None or did_direct or args.sweep):
        print("\nNothing to do. Provide either:")
        print("  --m0 & --mf   OR   --dry & --prop")
        print("  OR --target-dv (with --dry)")
        print("  OR --sweep (with --dry and sweep range).")


if __name__ == "__main__":
    main()
    # Default test run
    isp = 320      # s
    m0 = 50000     # kg (initial mass)
    mf = 10000     # kg (dry mass)
    g0 = 9.80665   # m/s^2

    # Δv from Tsiolkovsky rocket equation
    delta_v = isp * g0 * np.log(m0 / mf)

    # Example thrust calculation (assume mdot = (m0 - mf) / 100 sec burn)
    burn_time = 100.0
    mdot = (m0 - mf) / burn_time
    thrust = mdot * g0 * isp

    print("\n=== Test Run ===")
    print(f"Isp      : {isp:.1f} s")
    print(f"m0       : {m0:.1f} kg")
    print(f"mf (dry) : {mf:.1f} kg")
    print(f"Δv       : {delta_v:.2f} m/s ({delta_v/1000:.2f} km/s)")
    print(f"Thrust   : {thrust/1000:.2f} kN")

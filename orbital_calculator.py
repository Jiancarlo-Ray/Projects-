import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Earth constants
R_E = 6_371_000.0          # mean radius [m]
MU  = 3.986_004_418e14     # GM [m^3/s^2]

def compute_for_altitudes(altitudes_m):
    r = R_E + altitudes_m
    v = np.sqrt(MU / r)
    T = 2 * np.pi * np.sqrt(r**3 / MU)
    return v, T

def print_table(alts_m, v, T):
    print(f"{'Altitude (km)':>14} | {'Velocity (km/s)':>16} | {'Period (min)':>12}")
    print("-"*50)
    for h, vi, Ti in zip(alts_m, v, T):
        print(f"{h/1000:14.1f} | {vi/1000:16.3f} | {Ti/60:12.2f}")

def save_csv(filepath, alts_m, v, T):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["altitude_km", "velocity_km_s", "period_min"])
        for h, vi, Ti in zip(alts_m, v, T):
            writer.writerow([f"{h/1000:.1f}", f"{vi/1000:.3f}", f"{Ti/60:.2f}"])
    print(f"✓ CSV saved to {filepath}")

def make_plots(alts_m, v, T, outdir=None):
    alts_km = alts_m/1000
    plt.figure()
    plt.plot(alts_km, v/1000)
    plt.xlabel("Altitude (km)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Circular orbital velocity vs altitude")
    plt.grid(True)
    if outdir: 
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / "velocity_vs_altitude.png", dpi=200)
        print("✓ Saved velocity plot")
    else:
        plt.show()

    plt.figure()
    plt.plot(alts_km, T/60)
    plt.xlabel("Altitude (km)")
    plt.ylabel("Period (minutes)")
    plt.title("Orbital period vs altitude")
    plt.grid(True)
    if outdir: 
        plt.savefig(outdir / "period_vs_altitude.png", dpi=200)
        print("✓ Saved period plot")
    else:
        plt.show()

def parse_args():
    p = argparse.ArgumentParser(description="Orbital Mechanics Calculator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--alt", type=float, help="single altitude (km)")
    g.add_argument("--min", type=float, help="min altitude (km)")
    p.add_argument("--max", type=float, help="max altitude (km)")
    p.add_argument("--step", type=float, help="step size (km)")
    p.add_argument("--save", nargs="*", choices=["csv","plots"], help="save outputs")
    p.add_argument("--out", default="outputs", help="output folder")
    return p.parse_args()

def main():
    args = parse_args()
    if args.alt:
        alts_m = np.array([args.alt * 1000])
    else:
        if not args.max: 
            raise SystemExit("Error: --max required with --min")
        step = args.step if args.step else (args.max - args.min)/20
        alts_m = np.arange(args.min, args.max+step, step) * 1000

    v, T = compute_for_altitudes(alts_m)
    print_table(alts_m, v, T)

    outdir = Path(args.out)
    if args.save:
        if args.save and ("csv" in args.save or args.save == "csv"):
            save_csv(outdir / "orbital_results.csv", alts_m, v, T)
        if "plots" in args.save:
            make_plots(alts_m, v, T, outdir=outdir)
    else:
        if alts_m.size > 1:
            make_plots(alts_m, v, T)

if __name__ == "__main__":
    main()
    


import numpy as np

# Earth constants
R_E = 6_371_000.0          # mean radius [m]
MU  = 3.986_004_418e14     # GM [m^3/s^2]

def orbital_velocity(alt_km):
    #Returns circular orbital velocity at altitude (km)
    r = R_E + alt_km * 1000
    return np.sqrt(MU / r)   # [m/s]

def orbital_period(alt_km):
    #Returns orbital period at altitude (km)
    r = R_E + alt_km * 1000
    T = 2 * np.pi * np.sqrt(r**3 / MU)
    return T / 60   # [minutes]

if __name__ == "__main__":
    # Example altitudes
    for alt in [200, 400, 1000, 20000]:
        v = orbital_velocity(alt)
        T = orbital_period(alt)
        print(f"Altitude: {alt:>6} km | Velocity = {v/1000:.3f} km/s | Period = {T:.2f} min")

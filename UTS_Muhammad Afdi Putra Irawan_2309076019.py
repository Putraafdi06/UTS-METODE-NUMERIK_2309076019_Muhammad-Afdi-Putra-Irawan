import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 0.5  # Inductance in Henry
C = 10e-6  # Capacitance in Farad
target_f = 1000  # Target frequency in Hz
tolerance = 0.1  # Tolerance level for error in Ohms

# Function to calculate resonant frequency f as a function of resistance R
def f_R(R):
    term = 1 / (L * C) - (R**2) / (4 * L**2)
    if term <= 0:
        return None  # Invalid case if term inside sqrt is negative
    return (1 / (2 * np.pi)) * np.sqrt(term)

# Derivative of f(R) used in Newton-Raphson
def f_prime_R(R):
    term = 1 / (L * C) - (R**2) / (4 * L**2)
    if term <= 0:
        return None  # Invalid if term inside sqrt is negative
    sqrt_term = np.sqrt(term)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Newton-Raphson method
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None  # Return if f(R) is invalid
        error = f_val - target_f
        f_prime_val = f_prime_R(R)
        if f_prime_val is None:
            return None  # Return if derivative is invalid
        new_R = R - error / f_prime_val
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Bisection method
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        error = f_R(mid) - target_f
        if error is None:
            return None
        if abs(error) < tolerance:
            return mid
        if (f_R(a) - target_f) * error < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Run both methods
initial_guess = 50  # Initial guess for Newton-Raphson
interval_a, interval_b = 0, 100  # Range for bisection method

# Newton-Raphson results
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Not found"

# Bisection method results
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Not found"

# Display results
print("Newton-Raphson Method:")
print(f"Resistance: {R_newton} ohm, Frequency: {f_newton} Hz")

print("\nBisection Method:")
print(f"Resistance: {R_bisection} ohm, Frequency: {f_bisection} Hz")

# Plotting results
plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="red", linestyle="--", label="Target Frequency 1000 Hz")

# Plot Newton-Raphson results
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="blue", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

# Plot Bisection results
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="green", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

# Final plot formatting
plt.xlabel("Resistance R (Ohm)")
plt.ylabel("Resonant Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(True)
plt.show()
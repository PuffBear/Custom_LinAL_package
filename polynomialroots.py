from complexnumbers import ComplexNumber

def aberth_method(coefficients, max_iterations=1000, tolerance=1e-12):
    """Find roots of a polynomial using the Aberth method."""
    
    def evaluate_polynomial(z):
        """Evaluate the polynomial at a given point z."""
        result = ComplexNumber(0, 0)
        for i, coeff in enumerate(reversed(coefficients)):
            result += coeff * (z ** i)
        return result

    def evaluate_derivative(z):
        """Evaluate the derivative of the polynomial at a given point z."""
        result = ComplexNumber(0, 0)
        for i, coeff in enumerate(reversed(coefficients[1:]), start=1):
            result += coeff * i * (z ** (i - 1))
        return result

    def initialize_roots(n):
        roots = []
        for i in range(n):
            angle = 2 * i * ComplexNumber.pi() / n
            cos_part = ComplexNumber(ComplexNumber.cos(angle), 0)  # Wrap in ComplexNumber
            sin_part = ComplexNumber(ComplexNumber.sin(angle), 0) * ComplexNumber(0, 1)
            roots.append(cos_part + sin_part)  # Now both are ComplexNumber
        return roots

    n = len(coefficients) - 1  # Degree of the polynomial
    roots = initialize_roots(n)

    for _ in range(max_iterations):
        converged = True
        for i in range(n):
            zi = roots[i]
            p_val = evaluate_polynomial(zi)
            p_derivative = evaluate_derivative(zi)

            if p_derivative.real == 0 and p_derivative.imag == 0:
                raise ValueError("Derivative evaluated to zero; Aberth method cannot proceed.")

            # Compute the Aberth correction term
            correction = ComplexNumber(0, 0)
            for j in range(n):
                if i != j:
                    diff = zi - roots[j]
                    if diff.real == 0 and diff.imag == 0:
                        raise ValueError("Two roots are too close to each other.")
                    correction += ComplexNumber(1, 0) / diff
            
            # Update root using Aberth correction
            delta = p_val / p_derivative
            new_zi = zi - delta / (ComplexNumber(1, 0) - delta * correction)
            if abs(new_zi.real - zi.real) > tolerance or abs(new_zi.imag - zi.imag) > tolerance:
                converged = False

            roots[i] = new_zi

        if converged:
            break

    return roots


# Input the degree and coefficients
degree = int(input("Enter the degree of the polynomial: "))
coefficients = []
print("Enter the coefficients (real and imaginary parts for each):")
for _ in range(degree + 1):
    real = float(input("Real part: "))
    imag = float(input("Imaginary part: "))
    coefficients.append(ComplexNumber(real, imag))

# Compute roots
roots = aberth_method(coefficients)

# Output results
print("Roots of the polynomial:")
for root in roots:
    print(root)

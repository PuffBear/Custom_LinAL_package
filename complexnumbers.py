class ComplexNumber:
    def __init__(self, real, imag):
        if not isinstance(real, (int, float)) or not isinstance(imag, (int, float)):
            raise TypeError("Real and imaginary parts must be integers or floats.")
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imag)
        raise TypeError("Addition is only supported with ComplexNumber or scalar.")
    
    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        else:
            raise TypeError("Operand must be of type ComplexNumber.")

    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real
            )
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imag * other)
        raise TypeError("Multiplication is only supported with scalars or ComplexNumbers.")
    
    __rmul__ = __mul__  

    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            if other.real == 0 and other.imag == 0:
                raise ZeroDivisionError("Cannot divide by zero complex number.")
            denom = other.real ** 2 + other.imag ** 2
            real_part = (self.real * other.real + self.imag * other.imag) / denom
            imag_part = (self.imag * other.real - self.real * other.imag) / denom
            return ComplexNumber(real_part, imag_part)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return ComplexNumber(self.real / other, self.imag / other)
        else:
            raise TypeError("Operand must be of type ComplexNumber, int, or float.")

    def __abs__(self):
        return (self.real**2 + self.imag**2)**0.5

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def __neg__(self):
        return ComplexNumber(-self.real, -self.imag)

    def __str__(self):
        if self.imag < 0:
            return f"{self.real} - {-self.imag}i"
        return f"{self.real} + {self.imag}i"

    def __repr__(self):
        return f"ComplexNumber({self.real}, {self.imag})"

    def __eq__(self, other):
        return isinstance(other, ComplexNumber) and self.real == other.real and self.imag == other.imag
    
    def __pow__(self, power):
        if not isinstance(power, int):
            raise TypeError("Exponentiation is only supported for integer powers.")
        result = ComplexNumber(1, 0)
        base = self
        for _ in range(abs(power)):
            result *= base
        if power < 0:
            return ComplexNumber(1, 0) / result
        return result
    
    @staticmethod
    def pi():
        return 3.141592653589793

    @staticmethod
    def cos(angle):
        # Using Taylor series or similar approximation
        result = 1
        term = 1
        for n in range(1, 10):
            term *= (-1) * angle * angle / ((2 * n - 1) * (2 * n))
            result += term
        return result
    
    @staticmethod
    def sin(angle):
        # Using Taylor series or similar approximation
        result = angle
        term = angle
        for n in range(1, 10):
            term *= (-1) * angle * angle / ((2 * n) * (2 * n + 1))
            result += term
        return result
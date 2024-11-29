from complexnumbers import ComplexNumber

class Vector:
    def __init__(self, field_type, n, *coordinates):
        if field_type not in [float, int, ComplexNumber]:
            raise TypeError("Field type must be 'float', 'int', or 'ComplexNumber'.")
        
        if len(coordinates) != n:
            raise ValueError(f"Expected {n} coordinate values, but got {len(coordinates)}.")
        
        for value in coordinates:
            if not isinstance(value, field_type):
                raise TypeError(f"All coordinates must be of type {field_type}.")
        
        self.field_type = field_type
        self.length = n
        self.coordinates = list(coordinates)

    def _validate_same_type_and_length(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector.")
        if other.length != self.length:
            raise ValueError("Vectors must have the same length.")
        if other.field_type != self.field_type:
            raise TypeError("Vectors must have the same field type.")

    def __str__(self):
        return f"<{', '.join(map(str, self.coordinates))}>"

    def __repr__(self):
        commalist = ', '.join(repr(element) for element in self.coordinates)
        return f"{self.__class__.__name__}({commalist})"

    def __add__(self, other):
        self._validate_same_type_and_length(other)
        return Vector(self.field_type, self.length, *(a + b for a, b in zip(self.coordinates, other.coordinates)))
    
    __radd__ = __add__

    def __neg__(self):
        return Vector(self.field_type, self.length, *(-element for element in self.coordinates))
    
    def __sub__(self, other):
        self._validate_same_type_and_length(other)
        return Vector(self.field_type, self.length, *(a - b for a, b in zip(self.coordinates, other.coordinates)))
    
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            if other.length != self.length:
                raise ValueError("Vectors must have the same length for dot product.")
            if other.field_type != self.field_type:
                raise TypeError("Vectors must have the same field type for dot product.")
            return sum(a * b for a, b in zip(self.coordinates, other.coordinates))
        
        elif isinstance(other, (int, float, ComplexNumber)):
            if not isinstance(other, self.field_type) and not isinstance(other, (int, float)):
                raise TypeError(f"Scalar must be of type {self.field_type} or a compatible type (int, float).")
            return Vector(self.field_type, self.length, *(a * other for a in self.coordinates))
        
        else:
            raise TypeError("Multiplication is only supported with scalars or another vector.")

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Division only supported with scalars.")
        return Vector(self.field_type, self.length, *(a / scalar for a in self.coordinates))
    
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return (
            self.length == other.length and
            self.field_type == other.field_type and
            all(a == b for a, b in zip(self.coordinates, other.coordinates))
        )
    
    def __ne__(self, other):
        return not self == other
    
    def __abs__(self):
        return (sum(a * a for a in self.coordinates)) ** 0.5
    
    def vector_length(self):
        return sum(x * x for x in self.coordinates) ** 0.5
    
    def inner_product(self, other):
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError("Vectors must have the same dimension.")
        return sum(a * b for a, b in zip(self.coordinates, other.coordinates))
    
    def is_orthogonal(self, other):
        return self.inner_product(other) == 0
    
    
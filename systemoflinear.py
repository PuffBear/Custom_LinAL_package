from complexnumbers import ComplexNumber
from matrices import Matrix
from vectors import Vector

class LinearSystem:
    def __init__(self, A, b):
        if not isinstance(A, Matrix) or not isinstance(b, Vector):
            raise TypeError("A must be a Matrix and b must be a Vector.")
        
        if A.n != b.length:
            raise ValueError("Incompatible dimensions: A has {} rows but b has {} elements.".format(A.n, b.length))
        
        self.A = A  
        self.b = b  

    def is_consistent(self):
        # Create augmented matrix in a flattened format
        augmented_values = [value for i in range(self.A.n) for value in self.A.matrix[i] + [self.b.coordinates[i]]]
        augmented_matrix = Matrix(self.A.field_type, self.A.n, self.A.m + 1, *augmented_values)

        # Perform RREF on the augmented matrix
        rref_matrix = augmented_matrix.rref()

        # Check for inconsistency (row of all zeros except for last column)
        for row in rref_matrix.matrix:
            if all(val == 0 for val in row[:-1]) and row[-1] != 0:
                return False

        return True
  
    
    def solve(self):
        if not self.is_consistent():
            raise ValueError("The system is inconsistent and has no solutions.")

        # Create augmented matrix in a flattened format
        augmented_values = [value for i in range(self.A.n) for value in self.A.matrix[i] + [self.b.coordinates[i]]]
        augmented_matrix = Matrix(self.A.field_type, self.A.n, self.A.m + 1, *augmented_values)

        # Perform RREF
        rref_matrix = augmented_matrix.rref()

        # Extract the solution from the last column of RREF
        solution = [rref_matrix.matrix[i][-1] for i in range(self.A.m)]
        return Vector(self.A.field_type, self.A.m, *solution)

    
    @staticmethod
    def is_subspace(S1, S2):
        if not all(isinstance(v, Vector) for v in S1 + S2):
            raise TypeError("Both S1 and S2 must be lists of Vector objects.")
        
        for vector in S1:
            system = LinearSystem(
                Matrix(S2[0].field_type, len(S2[0].coordinates), len(S2), *sum([v.coordinates for v in S2], [])),
                vector
            )
            if not system.is_consistent():
                return False
        return True

    def solution_set_with_free_variables(self):
        # Flatten the augmented matrix data
        augmented_values = [value for i in range(self.A.n) for value in self.A.matrix[i] + [self.b.coordinates[i]]]
        augmented_matrix = Matrix(self.A.field_type, self.A.n, self.A.m + 1, *augmented_values)

        # Perform RREF on the augmented matrix
        rref_matrix = augmented_matrix.rref()

        # Identify pivot columns and free variables
        pivots = []
        free_vars = []
        for col in range(self.A.m):
            if any(rref_matrix.matrix[row][col] != 0 for row in range(len(rref_matrix.matrix))):
                pivots.append(col)
            else:
                free_vars.append(col)

        # Build solution set expressions
        solution = {}
        for var in free_vars:
            solution[f'x{var + 1}'] = 'free variable'

        for row, pivot_col in enumerate(pivots):
            # Check if the row exists in the RREF matrix
            if row < len(rref_matrix.matrix):
                expression = f"{rref_matrix.matrix[row][-1]}"
                for free_var in free_vars:
                    coefficient = -rref_matrix.matrix[row][free_var]
                    if coefficient != 0:
                        expression += f" + ({coefficient}) * x{free_var + 1}"
                solution[f'x{pivot_col + 1}'] = expression

        return solution

    
    def solve_with_plu(self):
        """Solve the system using PLU decomposition."""
        if not self.is_consistent():
            raise ValueError("The system is inconsistent and has no solutions.")
        
        P, L, U = self.A.plu_decomposition()
        Pb = Vector(self.b.field_type, self.b.length, *(sum(P.matrix[i][j] * self.b.coordinates[j] for j in range(self.b.length)) for i in range(self.b.length)))

        # Forward substitution to solve L * Y = P * b
        Y = [0] * self.A.n
        for i in range(self.A.n):
            Y[i] = Pb.coordinates[i] - sum(L.matrix[i][j] * Y[j] for j in range(i))
        
        # Backward substitution to solve U * X = Y
        X = [0] * self.A.m
        for i in reversed(range(self.A.m)):
            X[i] = (Y[i] - sum(U.matrix[i][j] * X[j] for j in range(i+1, self.A.m))) / U.matrix[i][i]
        
        return Vector(self.A.field_type, self.A.m, *X)
    
    def least_square_solution(self):
        if self.A.n < self.A.m:
            raise ValueError("The system is underdetermined; least square solution may not be unique.")
        A_pseudo = self.A.pseudoinverse()
        return A_pseudo * self.b

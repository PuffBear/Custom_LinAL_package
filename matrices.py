from complexnumbers import ComplexNumber
from vectors import Vector

class Matrix:
    def __init__(self, field_type, n, m, *values):
        if field_type not in [float, int, ComplexNumber]:
            raise ValueError("Field type must be 'float', 'int', or 'ComplexNumber'.")

        self.field_type = field_type
        self.n = n  
        self.m = m  

        if len(values) == m and all(isinstance(v, Vector) for v in values):
            for v in values:
                if v.length != n:
                    raise ValueError(f"All vectors must have length {n}.")
                if v.field_type != field_type:
                    raise TypeError(f"All vectors must have field type {field_type}.")

            self.matrix = [[v.coordinates[i] for v in values] for i in range(n)]

        elif len(values) == n * m:
            for value in values:
                if field_type == ComplexNumber and not isinstance(value, ComplexNumber):
                    raise TypeError("All values must be ComplexNumber instances.")
                elif field_type == float and not isinstance(value, (float, int)):
                    raise TypeError("All values must be floats.")
                elif field_type == int and not isinstance(value, int):
                    raise TypeError("All values must be integers.")

            self.matrix = [list(values[i * m:(i + 1) * m]) for i in range(n)]
        else:
            raise ValueError("Provide either n*m individual values or m vectors of length n.")
    
    def copy(self):
        """Create a deep copy of the matrix."""
        copied_matrix = Matrix(
            self.field_type,
            self.n,
            self.m,
            *[value for row in self.matrix for value in row]
        )
        return copied_matrix
    
    def __add__(self, other):
        if not isinstance(other, Matrix) or other.n != self.n or other.m != self.m:
            raise ValueError("Both matrices must have the same dimensions.")
        
        if self.field_type != other.field_type:
            raise TypeError("Matrix addition requires both matrices to have the same field type.")
        
        return Matrix(
            self.field_type,
            self.n,
            self.m,
            *(self.matrix[i][j] + other.matrix[i][j] for i in range(self.n) for j in range(self.m))
        )

    def __sub__(self, other):
        if not isinstance(other, Matrix) or other.n != self.n or other.m != self.m:
            raise ValueError("Both matrices must have the same dimensions.")
        
        if self.field_type != other.field_type:
            raise TypeError("Matrix subtraction requires both matrices to have the same field type.")

        return Matrix(
            self.field_type,
            self.n,
            self.m,
            *(self.matrix[i][j] - other.matrix[i][j] for i in range(self.n) for j in range(self.m))
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Handle scalar multiplication
            result = [
                self.matrix[i][j] * other
                for i in range(self.n)
                for j in range(self.m)
            ]
            return Matrix(self.field_type, self.n, self.m, *result)
        elif isinstance(other, Matrix):
            # Matrix-matrix multiplication
            if self.m != other.n:
                raise ValueError("Matrix multiplication requires the number of columns in the first matrix to match the number of rows in the second matrix.")
            result = []
            for i in range(self.n):
                row = []
                for j in range(other.m):
                    row.append(sum(self.matrix[i][k] * other.matrix[k][j] for k in range(self.m)))
                result.extend(row)
            return Matrix(self.field_type, self.n, other.m, *result)
        elif isinstance(other, Vector):
            # Matrix-vector multiplication
            if self.m != other.length:
                raise ValueError("Matrix-vector multiplication requires the number of columns in the matrix to match the vector's length.")
            result_coordinates = [
                sum(self.matrix[i][j] * other.coordinates[j] for j in range(self.m))
                for i in range(self.n)
            ]
            return Vector(self.field_type, self.n, *result_coordinates)
        else:
            raise TypeError("Matrix multiplication is only defined with another Matrix, Vector, or a scalar.")
        
    def __pow__(self, power):
        if not self.is_square():
            raise ValueError("Matrix exponentiation is only defined for square matrices.")
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer.")

        # Identity matrix
        result = Matrix(self.field_type, self.n, self.n, *[1 if i == j else 0 for i in range(self.n) for j in range(self.n)])
        
        if power == 0:
            return result

        temp = self
        while power > 0:
            if power % 2 == 1:
                result = result * temp  # Multiply current result by temp
            temp = temp * temp  # Square the matrix
            power //= 2

        return result
    
    @staticmethod
    def identity(field_type, size):
        return Matrix(field_type, size, size, *[1 if i == j else 0 for i in range(size) for j in range(size)])

    def get_row(self, row_index):
        if row_index < 0 or row_index >= self.n:
            raise IndexError("Row index out of bounds.")
        
        return Vector(self.field_type, self.m, *self.matrix[row_index])

    def get_column(self, col_index):
        if col_index < 0 or col_index >= self.m:
            raise IndexError("Column index out of bounds.")
        
        return Vector(self.field_type, self.n, *(self.matrix[i][col_index] for i in range(self.n)))

    def transpose(self):
        transposed_values = [[self.matrix[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(self.field_type, self.m, self.n, *[val for row in transposed_values for val in row])
    
    def conjugate(self):
        if self.field_type != ComplexNumber:
            raise TypeError("Conjugate operation is only defined for matrices with ComplexNumber elements.")
        
        conjugated_values = [[self.matrix[i][j].conjugate() for j in range(self.m)] for i in range(self.n)]
        return Matrix(self.field_type, self.n, self.m, *[val for row in conjugated_values for val in row])

    def transpose_conjugate(self):
        if self.field_type != ComplexNumber:
            raise TypeError("Transpose conjugate (Hermitian) is only defined for matrices with ComplexNumber elements.")
        
        return self.transpose().conjugate()

    def __abs__(self):
        return sum(abs(self.matrix[i][j])**2 for i in range(self.n) for j in range(self.m))**0.5

    def __str__(self):
        return "\n".join(["\t".join(map(str, row)) for row in self.matrix])

    def __eq__(self, other):
        if isinstance(other, Matrix) and self.n == other.n and self.m == other.m:
            return all(self.matrix[i][j] == other.matrix[i][j] for i in range(self.n) for j in range(self.m))
        return False
    
    def is_zero(self):
        return all(self.matrix[i][j] == 0 for i in range(self.n) for j in range(self.m))

    def is_square(self):
        return self.n == self.m

    def is_symmetric(self):
        if not self.is_square() or self.field_type != float:
            return False
        return all(self.matrix[i][j] == self.matrix[j][i] for i in range(self.n) for j in range(i, self.m))

    def is_hermitian(self):
        if not self.is_square() or self.field_type != ComplexNumber:
            return False
        return all(self.matrix[i][j] == self.matrix[j][i].conjugate() for i in range(self.n) for j in range(i, self.m))

    def is_orthogonal(self):
        if not self.is_square() or self.field_type != float:
            return False
        identity = Matrix(float, self.n, self.m, *(1 if i == j else 0 for i in range(self.n) for j in range(self.m)))
        return (self.transpose() * self) == identity

    def is_unitary(self):
        if self.field_type != ComplexNumber or not self.is_square():
            return False
        return self.transpose_conjugate() == self.inverse()

    def is_scalar(self):
        if not self.is_square():
            return False
        diagonal_value = self.matrix[0][0]
        return all(self.matrix[i][i] == diagonal_value for i in range(self.n)) and \
            all(self.matrix[i][j] == 0 for i in range(self.n) for j in range(self.m) if i != j)

    def is_singular(self):
        if not self.is_square():
            return False
        return self.determinant() == 0  # Assume a method to calculate determinant exists

    def is_invertible(self):
        if not self.is_square():
            return False
        return self.determinant() != 0  # Assume a method to calculate determinant exists

    def is_identity(self):
        if not self.is_square():
            return False
        return all(self.matrix[i][j] == (1 if i == j else 0) for i in range(self.n) for j in range(self.m))

    def is_nilpotent(self, k=None):
        if not self.is_square():
            return False
        current_power = self
        for _ in range(1, (k or self.n) + 1):
            current_power *= self
            if current_power.is_zero():
                return True
        return False

    def is_diagonalizable(self):
        return True

    def has_lu_decomposition(self):
        if not self.is_square():
            return False

        try:
            self.lu_decomposition()
            return True
        except ValueError:
            return False
        
    def has_plu_decomposition(self):
        if not self.is_square():
            return False

        try:
            self.plu_decomposition()
            return True
        except ValueError:
            return False

    
    def determinant(self):
        if not self.is_square():
            raise ValueError("Dxw eterminant is only defined for square matrices.")
        
        # Base case for 1x1 matrix
        if self.n == 1:
            return self.matrix[0][0]

        # Base case for 2x2 matrix
        if self.n == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

        # Recursive case for larger matrices
        det = 0
        for col in range(self.m):
            minor_matrix = self.get_minor(0, col)
            cofactor = (-1) ** col * self.matrix[0][col]
            det += cofactor * minor_matrix.determinant()
        
        return det

    def get_minor(self, row, col):
        minor_values = [self.matrix[i][j] for i in range(self.n) if i != row
                        for j in range(self.m) if j != col]
        return Matrix(self.field_type, self.n - 1, self.m - 1, *minor_values)
    
    def size(self):
        return (self.n, self.m)
    
    def rank(self):
        rref_matrix = self.rref()  
        non_zero_rows = sum(1 for row in rref_matrix.matrix if any(row))
        return non_zero_rows
    
    def nullity(self):
        # Nullity = number of columns - rank
        return self.size()[1] - self.rank()
    
    def rref(self, show_steps=False):
        rows, cols = self.size()
        mat = [row[:] for row in self.matrix]
        elementary_matrices = []  
        pivot_row = 0  

        for pivot_col in range(cols):
            if pivot_row >= rows:
                break
            
            # Step 1: Find a non-zero pivot in the current column
            max_row = max(range(pivot_row, rows), key=lambda r: abs(mat[r][pivot_col]))
            if mat[max_row][pivot_col] == 0:
                continue  # Move to the next column if the pivot column is zero
            
            if max_row != pivot_row:
                mat[pivot_row], mat[max_row] = mat[max_row], mat[pivot_row]
                if show_steps:
                    elementary_matrices.append(self.create_row_swap_matrix(rows, cols, pivot_row, max_row))

            # Step 2: Scale the pivot row to make the pivot element equal to 1
            pivot_element = mat[pivot_row][pivot_col]
            if pivot_element != 1:
                mat[pivot_row] = [x / pivot_element for x in mat[pivot_row]]
                if show_steps:
                    elementary_matrices.append(self.create_row_scale_matrix(rows, cols, pivot_row, 1 / pivot_element))

            # Step 3: Make all other entries in the pivot column zero
            for r in range(rows):
                if r != pivot_row and mat[r][pivot_col] != 0:
                    scale_factor = mat[r][pivot_col]
                    mat[r] = [a - scale_factor * b for a, b in zip(mat[r], mat[pivot_row])]
                    if show_steps:
                        elementary_matrices.append(self.create_row_replacement_matrix(rows, cols, r, pivot_row, -scale_factor))

            pivot_row += 1  

        if show_steps:
            for i, elem_matrix in enumerate(elementary_matrices, 1):
                print(f"Step {i}:")
                print(elem_matrix)
                print()

        # Update self.matrix with the RREF result
        self.matrix = mat
        self.clean_matrix()
        return self
    
    def clean_matrix(self):
        self.matrix = [[0.0 if abs(x) < 1e-10 else x for x in row] for row in self.matrix]


    # Helper functions for creating elementary matrices
    def create_identity_matrix(self, rows, cols):
        return [[1 if i == j else 0 for j in range(cols)] for i in range(rows)]

    def create_row_swap_matrix(self, rows, cols, row1, row2):
        identity = self.create_identity_matrix(rows, cols)
        identity[row1], identity[row2] = identity[row2], identity[row1]
        return Matrix(self.field_type, rows, cols, *[elem for row in identity for elem in row])

    def create_row_scale_matrix(self, rows, cols, row, scale):
        identity = self.create_identity_matrix(rows, cols)
        identity[row][row] = scale
        return Matrix(self.field_type, rows, cols, *[elem for row in identity for elem in row])

    def create_row_replacement_matrix(self, rows, cols, target_row, source_row, scale):
        identity = self.create_identity_matrix(rows, cols)
        identity[target_row][source_row] = scale
        return Matrix(self.field_type, rows, cols, *[elem for row in identity for elem in row])
    
    def are_vectors_linearly_independent(vectors):
        """ Determines if a list of vectors is linearly independent. """
        if not vectors:
            raise ValueError("The list of vectors cannot be empty.")

        n = vectors[0].length
        if any(vec.length != n for vec in vectors):
            raise ValueError("All vectors must have the same dimension.")

        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(n)]
        matrix = Matrix(float, n, len(vectors), *sum(matrix_data, []))

        rank = matrix.rank()

        return rank == len(vectors)
    
    def dimension_of_span(vectors):
        if not vectors:
            raise ValueError("The list of vectors cannot be empty.")
        
        vector_length = len(vectors[0].coordinates)
        if any(len(vec.coordinates) != vector_length for vec in vectors):
            raise ValueError("All vectors must have the same dimension.")
        
        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(vector_length)]
        matrix = Matrix(float, vector_length, len(vectors), *sum(matrix_data, []))
        
        return matrix.rank()
    
    def basis_for_span(vectors):
        if not vectors:
            raise ValueError("The list of vectors cannot be empty.")
        
        vector_length = vectors[0].length
        if any(vec.length != vector_length for vec in vectors):
            raise ValueError("All vectors must have the same dimension.")
        
        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(vector_length)]
        matrix = Matrix(float, vector_length, len(vectors), *sum(matrix_data, []))
        
        rref_matrix = matrix.rref()
        
        pivot_columns = []
        for i in range(rref_matrix.n):
            for j in range(rref_matrix.m):
                if rref_matrix.matrix[i][j] == 1:
                    pivot_columns.append(j)
                    break
        
        basis_vectors = [vectors[i] for i in pivot_columns]
        return basis_vectors
    
    def rank_factorization(self):        
        # Step 1: Perform RREF on the matrix to determine pivot columns
        rref_matrix = self.rref()
        pivot_columns = []
        
        for i in range(rref_matrix.n):
            for j in range(rref_matrix.m):
                if rref_matrix.matrix[i][j] == 1:
                    pivot_columns.append(j)
                    break
        
        # Step 2: Construct matrix C from the pivot columns of the original matrix
        C_data = []
        for j in pivot_columns:
            C_data.extend(self.get_column(j).coordinates)
        
        C = Matrix(self.field_type, self.n, len(pivot_columns), *C_data)

        # Step 3: Construct matrix F by expressing each column of A in terms of the pivot columns
        F_data = []
        for j in range(self.m):
            col_vec = [0] * len(pivot_columns)
            if j in pivot_columns:
                col_vec[pivot_columns.index(j)] = 1
            else:
                for i, pivot in enumerate(pivot_columns):
                    col_vec[i] = rref_matrix.matrix[i][j]
            F_data.extend(col_vec)
        
        F = Matrix(self.field_type, len(pivot_columns), self.m, *F_data)
        
        return C, F 
    
    def lu_decomposition(self):        
        # Step 1: Check if the matrix is square
        if self.n != self.m:
            raise ValueError("LU decomposition is only defined for square matrices.")
        
        L = Matrix(self.field_type, self.n, self.n, *[1 if i == j else 0 for i in range(self.n) for j in range(self.n)])  # Identity matrix for L
        U = Matrix(self.field_type, self.n, self.n, *sum(self.matrix, []))  # Copy of A for U
        
        for i in range(self.n):
            if U.matrix[i][i] == 0:
                raise ValueError("LU decomposition requires a non-zero pivot. Try using partial pivoting.")

            for j in range(i + 1, self.n):
                multiplier = U.matrix[j][i] / U.matrix[i][i]
                L.matrix[j][i] = multiplier  # Store the multiplier in L
                
                U.matrix[j] = [U.matrix[j][k] - multiplier * U.matrix[i][k] for k in range(self.n)]
        
        return L, U
    
    def plu_decomposition(self):        
        # Step 1: Check if the matrix is square
        if self.n != self.m:
            raise ValueError("PLU decomposition is only defined for square matrices.")
        
        P = Matrix(self.field_type, self.n, self.n, *[1 if i == j else 0 for i in range(self.n) for j in range(self.n)])  # Identity matrix for P
        L = Matrix(self.field_type, self.n, self.n, *[0 for _ in range(self.n * self.n)])  # Zero matrix for L
        U = Matrix(self.field_type, self.n, self.n, *sum(self.matrix, []))  # Copy of A for U
        
        for i in range(self.n):
            max_row = max(range(i, self.n), key=lambda r: abs(U.matrix[r][i]))
            if U.matrix[max_row][i] == 0:
                raise ValueError("Matrix is singular and cannot be decomposed with PLU.")
            
            if max_row != i:
                U.matrix[i], U.matrix[max_row] = U.matrix[max_row], U.matrix[i]
                P.matrix[i], P.matrix[max_row] = P.matrix[max_row], P.matrix[i]
                
                if i > 0:
                    L.matrix[i][:i], L.matrix[max_row][:i] = L.matrix[max_row][:i], L.matrix[i][:i]

            for j in range(i + 1, self.n):
                multiplier = U.matrix[j][i] / U.matrix[i][i]
                L.matrix[j][i] = multiplier  
                
                U.matrix[j] = [U.matrix[j][k] - multiplier * U.matrix[i][k] for k in range(self.n)]
        
        for i in range(self.n):
            L.matrix[i][i] = 1
        
        return P, L, U
    
    def inverse(self):
        # Step 1: Check if the matrix is square
        if not self.is_square():
            raise ValueError("Only square matrices can have an inverse.")
        
        # Step 2: Augment the matrix with the identity matrix
        identity_data = [1 if i == j else 0 for i in range(self.n) for j in range(self.n)]
        augmented_data = []
        for i in range(self.n):
            augmented_data.extend(self.matrix[i] + identity_data[i * self.n:(i + 1) * self.n])
        augmented_matrix = Matrix(self.field_type, self.n, 2 * self.n, *augmented_data)
        
        # Step 3: Perform Gaussian elimination to reduce the left side to identity
        for i in range(self.n):
            # Ensure pivot is not zero
            if augmented_matrix.matrix[i][i] == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")
            
            # Scale the row to make the pivot 1
            pivot = augmented_matrix.matrix[i][i]
            augmented_matrix.matrix[i] = [x / pivot for x in augmented_matrix.matrix[i]]
            
            # Eliminate all other entries in the pivot column
            for j in range(self.n):
                if i != j:  # Skip the pivot row
                    factor = augmented_matrix.matrix[j][i]
                    augmented_matrix.matrix[j] = [
                        augmented_matrix.matrix[j][k] - factor * augmented_matrix.matrix[i][k]
                        for k in range(2 * self.n)
                    ]
        
        # Step 4: Extract the right side as the inverse
        inverse_values = [row[self.n:] for row in augmented_matrix.matrix]
        flattened_inverse = [val for row in inverse_values for val in row]
        return Matrix(self.field_type, self.n, self.n, *flattened_inverse)

    
    def inverse_by_adjoint(self):
        # Step 1: Check if the matrix is square
        if not self.is_square():
            raise ValueError("Inverse is only defined for square matrices.")

        # Step 2: Compute the determinant
        determinant = self.determinant()
        if determinant == 0:
            return "Matrix is not invertible."

        # Step 3: Compute the cofactor matrix
        cofactor_values = []
        for row in range(self.n):
            for col in range(self.m):
                # Compute the minor matrix for each element
                minor = self.get_minor(row, col)
                cofactor = ((-1) ** (row + col)) * minor.determinant()
                cofactor_values.append(cofactor)

        # Step 4: Form the cofactor matrix
        cofactor_matrix = Matrix(self.field_type, self.n, self.m, *cofactor_values)

        # Step 5: Compute the adjoint by transposing the cofactor matrix
        adjoint_matrix = cofactor_matrix.transpose()

        # Step 6: Scale the adjoint matrix by 1 / determinant
        inverse_values = [value / determinant for row in adjoint_matrix.matrix for value in row]
        inverse_matrix = Matrix(self.field_type, self.n, self.m, *inverse_values)

        return inverse_matrix 
    
    def determinant_cofactor(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices.")
        
        if self.n == 1:  # Base case for 1x1 matrix
            return self.matrix[0][0]
        
        if self.n == 2:  # Base case for 2x2 matrix
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

        # Recursive cofactor expansion along the first row
        det = 0
        for col in range(self.n):
            cofactor = (-1) ** col * self.matrix[0][col]
            minor_matrix = self.get_minor(0, col)
            det += cofactor * minor_matrix.determinant_cofactor()
        
        return det
    
    def determinant_plu(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices.")
        
        P, L, U = self.plu_decomposition()
        det_P = (-1) ** sum(P.matrix[i][j] != (1 if i == j else 0) for i in range(P.n) for j in range(P.m))  # Handle permutations
        det_L = 1  # L is lower triangular with 1's on the diagonal
        det_U = 1
        for i in range(U.n):
            det_U *= U.matrix[i][i]  # Product of U's diagonal elements
        
        return det_P * det_L * det_U
    
    def determinant_rref(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices.")
        
        mat = [row[:] for row in self.matrix]  # Copy the matrix
        scaling_factor = 1
        transpositions = 0

        for i in range(self.n):
            # Pivot selection
            pivot_row = max(range(i, self.n), key=lambda r: abs(mat[r][i]))
            if mat[pivot_row][i] == 0:
                return 0  # Singular matrix

            if pivot_row != i:
                mat[i], mat[pivot_row] = mat[pivot_row], mat[i]
                transpositions += 1

            # Scale pivot row
            scaling_factor *= mat[i][i]
            mat[i] = [x / mat[i][i] for x in mat[i]]

            # Eliminate below pivot
            for r in range(i + 1, self.n):
                factor = mat[r][i]
                mat[r] = [a - factor * b for a, b in zip(mat[r], mat[i])]

        determinant = scaling_factor * (-1) ** transpositions
        return determinant
    
    def qr_decomposition(self):
        if not self.is_square():
            raise ValueError("QR decomposition is only implemented for square matrices.")

        n = self.n  # Number of rows/columns for square matrix
        Q_columns = []
        R = [[0] * n for _ in range(n)]  # Initialize R as a zero matrix

        for i in range(n):
            # Get the i-th column of the matrix as a vector
            v = [self.matrix[row][i] for row in range(n)]

            # Perform Gram-Schmidt orthogonalization
            for j in range(len(Q_columns)):
                q = Q_columns[j]
                R[j][i] = sum(v[k] * q[k] for k in range(n))  # Dot product of v and q
                v = [v[k] - R[j][i] * q[k] for k in range(n)]  # Subtract projection

            # Normalize v to get the i-th column of Q
            norm_v = sum(x ** 2 for x in v) ** 0.5
            if norm_v == 0:
                raise ValueError("Matrix is rank-deficient; QR decomposition failed.")
            q_i = [x / norm_v for x in v]
            Q_columns.append(q_i)

            # Set the diagonal entry of R
            R[i][i] = norm_v

        # Construct Q from Q_columns
        Q = Matrix(self.field_type, n, n, *[Q_columns[j][i] for i in range(n) for j in range(n)])

        # Construct R from its computed values
        R_matrix = Matrix(self.field_type, n, n, *sum(R, []))

        return Q, R_matrix
    
    def eigenvalues(self, max_iterations=1000, tolerance=1e-12):
        if not self.is_square():
            raise ValueError("Eigenvalues are only defined for square matrices.")
        
        A = self.copy()  # Copy the matrix to preserve the original
        n = A.n

        for _ in range(max_iterations):
            # Step 1: QR Decomposition
            Q, R = A.qr_decomposition()

            # Step 2: Update A
            A = R * Q  # Multiply R and Q to form A_{k+1}

            # Step 3: Check for convergence
            off_diagonal_sum = sum(abs(A.matrix[i][j]) for i in range(n) for j in range(i))
            if off_diagonal_sum < tolerance:
                break

        # Step 4: Return eigenvalues (diagonal elements)
        return [A.matrix[i][i] for i in range(n)]
    
    def eigenvectors(self, eigenvalues, tolerance=1e-10):
        #placeholder
        return

    def characteristic_polynomial(matrix):
        if not matrix.is_square():
            raise ValueError("Characteristic polynomial is only defined for square matrices.")
        
        n = matrix.n  # Size of the matrix

        # Step 1: Use QR decomposition to approximate eigenvalues
        eigenvalues = matrix.eigenvalues()  # Assuming eigenvalues() is implemented

        # Step 2: Construct the characteristic polynomial
        # P(λ) = (λ - λ_1)(λ - λ_2)...(λ - λ_n)
        coefficients = [1]  # Start with leading coefficient of 1 (monic polynomial)
        for eigenvalue in eigenvalues:
            new_coefficients = [0] * (len(coefficients) + 1)
            for i in range(len(coefficients)):
                new_coefficients[i] -= coefficients[i] * eigenvalue.real
                new_coefficients[i + 1] += coefficients[i]
            coefficients = new_coefficients

        return coefficients
    
    def minimal_polynomial(matrix):
        if not matrix.is_square():
            raise ValueError("Minimal polynomial is only defined for square matrices.")
        
        n = matrix.n  # Size of the matrix
        def generate_polynomial_coefficients(degree):
            def recursive_generate(current, depth):
                if depth == 0:
                    yield current
                else:
                    for coeff in range(-10, 11):  # Example range for coefficients
                        yield from recursive_generate(current + [coeff], depth - 1)
            
            return list(recursive_generate([], degree + 1))

        # Step 1: Use eigenvalues to determine potential roots
        eigenvalues = matrix.eigenvalues()  # Assuming eigenvalues() is implemented
        unique_eigenvalues = list(set(eigenvalues))

        # Step 2: Test candidate polynomials
        def evaluate_polynomial_at_matrix(coefficients):
            result = Matrix.identity(matrix.field_type, n)
            for i, coeff in enumerate(coefficients):
                result += matrix ** i * coeff
            return result

        for degree in range(1, n + 1):
            # Generate all combinations of coefficients for polynomials of this degree
            coefficients = generate_polynomial_coefficients(degree)  # Implemented as a helper
            for poly in coefficients:
                poly_matrix = evaluate_polynomial_at_matrix(poly)
                if poly_matrix.is_zero():  # If polynomial annihilates matrix
                    return poly

        raise ValueError("Failed to find minimal polynomial.")
    
    def eigenvectors(self, eigenvalue):
        if not self.is_square():
            raise ValueError("Eigenvectors can only be computed for square matrices.")
        
        # Step 1: Form (A - λI)
        shifted_matrix = self - Matrix.identity(self.field_type, self.n) * eigenvalue

        # Step 2: Compute RREF of (A - λI)
        rref_result = shifted_matrix.rref()

        # Step 3: Determine eigenvectors from RREF
        eigenvectors = []
        free_variables = [True] * self.n  # Initially assume all variables are free

        for row in rref_result.matrix:
            for j in range(len(row)):
                if row[j] != 0:  # Pivot column
                    free_variables[j] = False

        # Step 4: Generate eigenvectors for free variables
        for i in range(self.n):
            if free_variables[i]:  # Generate vector for each free variable
                eigenvector_coords = [0] * self.n
                eigenvector_coords[i] = 1
                for row_idx, row in enumerate(rref_result.matrix):
                    if row[i] != 0:  # Use row to adjust free variable
                        eigenvector_coords[row_idx] = -row[i]
                eigenvectors.append(Vector(self.field_type, self.n, *eigenvector_coords))

        return eigenvectors



    def null_space(self):
        if not self.is_square():
            raise ValueError("Null space computation requires a square matrix.")
        
        # Perform RREF on the matrix
        rref_matrix = self.rref()

        n, m = self.n, self.m
        pivot_cols = []
        free_cols = []

        # Identify pivot and free columns
        for col in range(m):
            if any(rref_matrix.matrix[row][col] != 0 for row in range(n)):
                pivot_cols.append(col)
            else:
                free_cols.append(col)

        # Generate null space basis
        null_space_basis = []
        for free_col in free_cols:
            vector = [0] * m
            vector[free_col] = 1
            for row, pivot_col in enumerate(pivot_cols):
                vector[pivot_col] = -rref_matrix.matrix[row][free_col]
            null_space_basis.append(vector)

        return null_space_basis
    
    def algebraic_multiplicity(self, eigenvalue):
        char_poly = self.characteristic_polynomial()
        multiplicity = 0
        for coeff in char_poly:
            if abs(coeff - eigenvalue) < 1e-6:  # Compare with a small tolerance
                multiplicity += 1
        return multiplicity

    def geometric_multiplicity(self, eigenvalue):
        if not self.is_square():
            raise ValueError("Geometric multiplicity is only defined for square matrices.")
        
        # Form (A - λI)
        n = self.n
        lambda_identity = Matrix(self.field_type, n, n, *[eigenvalue if i == j else 0 for i in range(n) for j in range(n)])
        A_minus_lambda_I = self - lambda_identity

        # Compute the rank of (A - λI)
        rank = A_minus_lambda_I.rank()

        # Geometric multiplicity is n - rank
        return n - rank

    def eigen_basis(self, eigenvalue):
        if not self.is_square():
            raise ValueError("Eigen-basis is only defined for square matrices.")
        
        # Form (A - λI)
        n = self.n
        lambda_identity = Matrix(self.field_type, n, n, *[eigenvalue if i == j else 0 for i in range(n) for j in range(n)])
        A_minus_lambda_I = self - lambda_identity

        # Compute the null space of (A - λI)
        null_space = A_minus_lambda_I.null_space()
        return null_space
    
    def is_diagonalizable(self):
        """Check if the matrix is diagonalizable."""
        if not self.is_square():
            raise ValueError("Diagonalizability is only defined for square matrices.")
        
        eigenvalues = self.eigenvalues()
        total_geometric_multiplicity = sum(self.geometric_multiplicity(eigenvalue) for eigenvalue in eigenvalues)
        
        return total_geometric_multiplicity == self.n  # Diagonalizable if total geometric multiplicities equal matrix size

    def diagonalization_basis(self):
        """Find the change of basis matrix to diagonalize the matrix."""
        if not self.is_square():
            raise ValueError("Diagonalization is only defined for square matrices.")
        if not self.is_diagonalizable():
            raise ValueError("The matrix is not diagonalizable.")

        eigenvalues = self.eigenvalues()
        basis_vectors = []

        for eigenvalue in eigenvalues:
            basis = self.eigen_basis(eigenvalue)  # Get eigenvectors for each eigenvalue
            basis_vectors.extend(basis)
        
        # Form the change of basis matrix P
        flattened_basis = [coord for vector in basis_vectors for coord in vector.coordinates]
        P = Matrix(self.field_type, self.n, self.n, *flattened_basis)

        return P
    
    def pseudoinverse(self):
        U, S, V_T = self.singular_value_decomposition()  # Assuming SVD implementation exists
        S_inv = Matrix(self.field_type, S.n, S.m, *(1 / s if s > 1e-10 else 0 for s in S.diagonal()))
        return V_T.transpose() * S_inv * U.transpose()
    
    def sqrt(self):
        if not self.is_square():
            raise ValueError("Matrix square root is only defined for square matrices.")

        # Step 1: Compute eigenvalues and eigenvectors
        eigenvalues = self.eigenvalues()  # Ensure this method works as expected
        eigenvectors = self.eigenvectors(eigenvalues)

        # Step 2: Compute the square root of eigenvalues
        sqrt_eigenvalues = [e**0.5 if e > 0 else 0 for e in eigenvalues]  # Handle only non-negative eigenvalues

        # Step 3: Form the diagonal matrix of square roots of eigenvalues
        sqrt_diagonal_data = [0] * (self.n * self.n)
        for i in range(len(sqrt_eigenvalues)):
            sqrt_diagonal_data[i * self.n + i] = sqrt_eigenvalues[i]
        sqrt_diagonal = Matrix(self.field_type, self.n, self.n, *sqrt_diagonal_data)

        # Step 4: Form the square root matrix: Q * Λ^(1/2) * Q^T
        Q = Matrix(self.field_type, self.n, self.n, *[v.coordinates for v in eigenvectors])
        return Q * sqrt_diagonal * Q.transpose()


    
    def polar_decomposition(self):
        if self.n != self.m:
            raise ValueError("Polar decomposition is only defined for square matrices.")

        # Step 1: Compute A^T * A (real case, A* for complex case)
        A_T_A = self.transpose() * self

        # Step 2: Compute the square root of A^T * A to get P
        P = A_T_A.sqrt()  # Assume a method to compute the square root of a positive definite matrix

        # Step 3: Compute the inverse of P
        P_inv = P.inverse()

        # Step 4: Compute U = A * P_inv
        U = self * P_inv

        return U, P
    

from vectors import Vector
from matrices import Matrix
from systemoflinear import LinearSystem

class VectorSpace:
    def __init__(self, field_type, vectors):
        if not all(isinstance(v, Vector) for v in vectors):
            raise TypeError("All elements must be of type Vector.")
        if len(vectors) == 0:
            raise ValueError("The vector set cannot be empty.")
        if not all(v.field_type == field_type for v in vectors):
            raise TypeError("All vectors must have the same field type.")
        if not all(v.length == vectors[0].length for v in vectors):
            raise ValueError("All vectors must have the same dimension.")

        self.field_type = field_type
        self.vectors = vectors
        self.dimension = vectors[0].length

    def is_in_span(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("The provided input must be a Vector.")
        if vector.length != self.dimension:
            raise ValueError("The vector must have the same dimension as the set.")

        # Step 1: Form a matrix where columns are the vectors in the set
        matrix_data = []
        for vec in self.vectors:
            matrix_data.extend(vec.coordinates)
        matrix = Matrix(self.field_type, self.dimension, len(self.vectors), *matrix_data)

        # Step 2: Append the given vector as the last column
        augmented_data = matrix_data + vector.coordinates
        augmented_matrix = Matrix(self.field_type, self.dimension, len(self.vectors) + 1, *augmented_data)

        # Step 3: Perform row reduction on the augmented matrix
        rref_matrix = augmented_matrix.rref()

        # Step 4: Check if the system is consistent
        # If the last column of the RREF contains a pivot in any row where the coefficient matrix has none,
        # the system is inconsistent.
        for i in range(self.dimension):
            if all(rref_matrix.matrix[i][j] == 0 for j in range(len(self.vectors))) and rref_matrix.matrix[i][-1] != 0:
                return False

        return True
    
    def representation_in_span(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("The provided input must be a Vector.")
        if vector.length != self.dimension:
            raise ValueError("The vector must have the same dimension as the set.")

        # Step 1: Form a matrix where columns are the vectors in the set
        matrix_data = []
        for vec in self.vectors:
            matrix_data.extend(vec.coordinates)
        A = Matrix(self.field_type, self.dimension, len(self.vectors), *matrix_data)

        # Step 2: Use the vector as the right-hand side
        b = vector

        # Step 3: Solve the system Ax = b using LinearSystem
        system = LinearSystem(A, b)
        if not system.is_consistent():
            raise ValueError("The vector is not in the span of the given set of vectors.")

        # Step 4: Extract solution
        solution = system.solve()
        return solution.coordinates
    
    def spans_same_subspace(S1, S2):
        if not all(isinstance(v, Vector) for v in S1 + S2):
            raise TypeError("Both S1 and S2 must be lists of Vector objects.")
        if not all(v.length == S1[0].length for v in S1 + S2):
            raise ValueError("All vectors must have the same dimension.")

        # Check if every vector in S1 is in the span of S2
        for vec in S1:
            if not VectorSpace(S2[0].field_type, S2).is_in_span(vec):
                return False

        # Check if every vector in S2 is in the span of S1
        for vec in S2:
            if not VectorSpace(S1[0].field_type, S1).is_in_span(vec):
                return False

        return True
    
    def coordinates_in_basis(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("The input must be a Vector.")
        if vector.length != self.dimension:
            raise ValueError("The vector must have the same dimension as the basis.")

        # Form matrix A with basis vectors as columns
        matrix_data = [vec.coordinates for vec in self.vectors]
        flattened_data = [val for col in matrix_data for val in col]
        A = Matrix(self.field_type, self.dimension, len(self.vectors), *flattened_data)

        # Solve Ax = vector to find coordinates
        system = LinearSystem(A, vector)
        if not system.is_consistent():
            raise ValueError("The vector is not in the span of the basis.")

        solution = system.solve()
        return solution.coordinates

    def vector_from_coordinates(self, coordinates):
        if len(coordinates) != len(self.vectors):
            raise ValueError("Number of coordinates must match the number of basis vectors.")

        # Reconstruct vector by linear combination of basis vectors
        reconstructed_coordinates = [0] * self.dimension
        for scalar, basis_vector in zip(coordinates, self.vectors):
            for i in range(self.dimension):
                reconstructed_coordinates[i] += scalar * basis_vector.coordinates[i]

        return Vector(self.field_type, self.dimension, *reconstructed_coordinates)
    
    @staticmethod
    def change_of_basis(B1, B2):
        if len(B1) != len(B2):
            raise ValueError("Both bases must have the same number of vectors.")
        if not all(v.length == B1[0].length for v in B1 + B2):
            raise ValueError("All vectors in both bases must have the same dimension.")
        if not all(v.field_type == B1[0].field_type for v in B1 + B2):
            raise TypeError("All vectors in both bases must have the same field type.")

        field_type = B1[0].field_type
        dimension = B1[0].length

        # Form a matrix where columns are the vectors in B2
        B2_matrix_data = [vec.coordinates for vec in B2]
        B2_flattened = [val for col in B2_matrix_data for val in col]
        B2_matrix = Matrix(field_type, dimension, len(B2), *B2_flattened)

        # Initialize the list to hold the coordinates of B1 vectors in B2
        change_of_basis_data = []

        for b1_vec in B1:
            # Solve B2_matrix * x = b1_vec
            system = LinearSystem(B2_matrix, b1_vec)
            if not system.is_consistent():
                raise ValueError("B1 and B2 do not span the same space.")
            solution = system.solve()

            # Append the solution (coordinates) to the change of basis matrix
            change_of_basis_data.extend(solution.coordinates)

        # Return the change of basis matrix
        return Matrix(field_type, len(B1), len(B2), *change_of_basis_data)
    
    @staticmethod
    def coordinates_in_new_basis(B1, B2, coordinates_B1):
        if len(B1) != len(B2):
            raise ValueError("B1 and B2 must have the same number of vectors.")
        if len(coordinates_B1) != len(B1):
            raise ValueError("The number of coordinates must match the number of vectors in B1.")
        
        # Construct the B1_matrix
        field_type = B1[0].field_type
        dimension = B1[0].length

        B1_matrix_data = [vec.coordinates for vec in B1]
        B1_flattened = [val for col in B1_matrix_data for val in col]
        B1_matrix = Matrix(field_type, dimension, len(B1), *B1_flattened)

        # Construct the B2_matrix
        B2_matrix_data = [vec.coordinates for vec in B2]
        B2_flattened = [val for col in B2_matrix_data for val in col]
        B2_matrix = Matrix(field_type, dimension, len(B2), *B2_flattened)

        # Calculate the change-of-basis matrix: P_B1_to_B2 = B2_matrix @ B1_matrix.inverse()
        try:
            B1_matrix_inv = B1_matrix.inverse()
        except ValueError as e:
            raise ValueError("The matrix B1 is not invertible.") from e

        P_B1_to_B2 = B2_matrix * B1_matrix_inv

        # Transform the coordinates in B1 to coordinates in B2
        coordinates_B1_vector = Vector(field_type, len(B1), *coordinates_B1)
        coordinates_B2 = P_B1_to_B2 * coordinates_B1_vector

        return coordinates_B2.coordinates
    
    def gram_schmidt(self):
        orthogonal_vectors = []
        for v in self.vectors:
            orthogonal_component = v
            for u in orthogonal_vectors:
                projection = u * (v.inner_product(u) / u.inner_product(u))
                orthogonal_component -= projection
            orthogonal_vectors.append(orthogonal_component)

        # Normalize the orthogonal vectors
        orthonormal_basis = [v / abs(v) for v in orthogonal_vectors if abs(v) > 0]
        return orthonormal_basis

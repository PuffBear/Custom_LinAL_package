# Project Code Structure

This README provides an overview of the classes and functions defined in the project files.

## File: complexnumbers.py
# ComplexNumber Class

## Overview

The `ComplexNumber` class provides a representation of complex numbers and implements basic arithmetic operations, as well as commonly used methods like absolute value and conjugate. This class overloads Python operators for seamless integration with arithmetic expressions involving complex numbers.

---

## Features

1. **Arithmetic Operations**:
   - Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`)
   - Handles operations with both complex numbers and scalars.

2. **Mathematical Methods**:
   - Absolute Value (`abs()`): Calculates the magnitude of the complex number.
   - Conjugate: Returns the complex conjugate.

3. **Utility Functions**:
   - String Representation (`__str__`): Provides a human-readable format.
   - Developer-Friendly Representation (`__repr__`): Suitable for debugging.

4. **Equality Check**:
   - Compares two complex numbers for equality.

---

## Class Definition

### Constructor

ComplexNumber(real, imag)

### Arguments:

real  (int/float): The real part of the complex number.

imag  (int/float): The imaginary part of the complex number.

z = ComplexNumber(3, 4)  # Represents 3 + 4i


## Methods and Overloaded Operators

### Arithmetic Operations

#### **Addition (`__add__`)**
- Adds two complex numbers.
- **Usage**: `z1 + z2`
- **Arguments**:
  - `other` (ComplexNumber): The complex number to add.
- **Example**:

  z1 = ComplexNumber(3, 4)

  z2 = ComplexNumber(1, -2)

  result = z1 + z2  # result = ComplexNumber(4, 2)

#### **Subtraction (`__sub__`)**
- Subtracts two complex numbers.
- **Usage**: `z1 - z2`
- **Example**:
  
  result = z1 - z2  # result = ComplexNumber(2, 6)

#### **Multiplication (`__mul__`)**
- Multiplies two complex numbers or a complex number by a scalar.
- **Usage**: `z1 * z2` or `z1 * scalar`
- **Example**:
  
  result = z1 * z2  # result = ComplexNumber(11, 2)

#### **Division (`__truediv__`)**
- Divides one complex number by another or by a scalar.
- **Usage**: `z1 / z2` or `z1 / scalar`
- **Example**:
  
  result = z1 / z2

### Mathematical Methods

#### **Absolute Value (`__abs__`)**
- Computes the magnitude of the complex number.
- **Usage**: `abs(z)`
- **Example**:
  
  z = ComplexNumber(3, 4)

  print(abs(z))  # Outputs: 5.0

#### **Conjugate (`conjugate`)**
- Returns the complex conjugate.
- **Usage**: `z.conjugate()`
- **Example**:
  
  z = ComplexNumber(3, 4)

  print(z.conjugate())  # Outputs: ComplexNumber(3, -4)

#### **Equality (`__eq__`)**
- Checks if two complex numbers are equal.
- **Usage**: `z1 == z2`
- **Example**:
  
  z1 = ComplexNumber(3, 4)

  z2 = ComplexNumber(3, 4)

  print(z1 == z2)  # True

#### **String Representations**
**(`__str__`)**
- Converts the complex number to a human-readable string.
- **Example**:
  
  z = ComplexNumber(3, 4)
  print(z)  # "3 + 4i"

**(`__repr__`)**
- Returns a developer-friendly string.
- **Example**:
  
  z = ComplexNumber(3, 4)

  repr(z)  # "ComplexNumber(3, 4)"


## Notes:
- This class is designed to handle arithmetic operations with both **(`ComplexNumber`)** instances and scalars **(`int`/`float`)**
- Error messages clearly describe invalid operations.



# Matrix Class: Mathematical and Utility Methods

The `Matrix` class provides a comprehensive representation of matrices and includes methods for performing essential mathematical operations, such as addition, subtraction, multiplication, and power. It also offers utility functions to manipulate and analyze matrices, including methods for obtaining the transpose, conjugate, and determinant, as well as checks for specific properties like symmetry and invertibility. This class supports seamless integration with Python operators and is designed for both beginner and advanced matrix operations.

# Features

## Initialization (`__init__`)
- Initializes a `Matrix` object with given rows or a list of lists.
- **Usage**: `Matrix(rows)`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m)  # Outputs: Matrix([[1, 2], [3, 4]])
  

## Copy (`copy`)
- Creates a deep copy of the matrix.
- **Usage**: `matrix.copy()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  m_copy = m.copy()

  print(m_copy)  # Outputs: Matrix([[1, 2], [3, 4]])

  print(m is m_copy)  # Outputs: False (Different objects)
  

## Addition (`__add__`)
- Adds two matrices of the same dimensions element-wise.
- **Usage**: `matrix1 + matrix2`
- **Example**:
  
  m1 = Matrix([[1, 2], [3, 4]])

  m2 = Matrix([[5, 6], [7, 8]])

  result = m1 + m2

  print(result)  # Outputs: Matrix([[6, 8], [10, 12]])
  

## Subtraction (`__sub__`)
- Subtracts one matrix from another element-wise.
- **Usage**: `matrix1 - matrix2`
- **Example**:
  
  m1 = Matrix([[5, 6], [7, 8]])

  m2 = Matrix([[1, 2], [3, 4]])

  result = m1 - m2

  print(result)  # Outputs: Matrix([[4, 4], [4, 4]])
  

## Multiplication (`__mul__`)
- Multiplies two matrices if their dimensions are compatible or scales by a scalar.
- **Usage**: `matrix1 * matrix2` or `matrix * scalar`
- **Example**:
  
  m1 = Matrix([[1, 2], [3, 4]])

  m2 = Matrix([[2, 0], [1, 2]])

  result = m1 * m2

  print(result)  # Outputs: Matrix([[4, 4], [10, 8]])
  
  scalar_result = m1 * 2

  print(scalar_result)  # Outputs: Matrix([[2, 4], [6, 8]])
  

## Power (`__pow__`)
- Computes the power of a square matrix.
- **Usage**: `matrix ** n`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  result = m ** 2

  print(result)  # Outputs: Matrix([[7, 10], [15, 22]])
  

## Identity Matrix (`identity`)
- Returns the identity matrix of size `n`.
- **Usage**: `Matrix.identity(n)`
- **Example**:

  id_matrix = Matrix.identity(3)

  print(id_matrix)  # Outputs: Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

## Transpose (`transpose`)
- Returns the transpose of the matrix.
- **Usage**: `matrix.transpose()`
- **Example**:
  
  m = Matrix([[1, 2, 3], [4, 5, 6]])

  transposed = m.transpose()

  print(transposed)  # Outputs: Matrix([[1, 4], [2, 5], [3, 6]])
  

## Equality (`__eq__`)
- Checks if two matrices are equal.
- **Usage**: `matrix1 == matrix2`
- **Example**:
  
  m1 = Matrix([[1, 2], [3, 4]])

  m2 = Matrix([[1, 2], [3, 4]])

  print(m1 == m2)  # Outputs: True
  

## Zero Matrix Check (`is_zero`)
- Checks if the matrix is a zero matrix (all elements are 0).
- **Usage**: `matrix.is_zero()`
- **Example**:
  
  m = Matrix([[0, 0], [0, 0]])

  print(m.is_zero())  # Outputs: True
  

## Square Matrix Check (`is_square`)
- Checks if the matrix is square (number of rows equals number of columns).
- **Usage**: `matrix.is_square()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.is_square())  # Outputs: True
  
  m_non_square = Matrix([[1, 2, 3], [4, 5, 6]])

  print(m_non_square.is_square())  # Outputs: False
  

## Symmetric Matrix Check (`is_symmetric`)
- Checks if the matrix is symmetric (equal to its transpose).
- **Usage**: `matrix.is_symmetric()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 1]])

  print(m.is_symmetric())  # Outputs: True
  

## Hermitian Matrix Check (`is_hermitian`)
- Checks if the matrix is Hermitian (equal to its conjugate transpose).
- **Usage**: `matrix.is_hermitian()`
- **Example**:
  
  m = Matrix([[1, 2j], [-2j, 1]])

  print(m.is_hermitian())  # Outputs: True
  

## Orthogonal Matrix Check (`is_orthogonal`)
- Checks if the matrix is orthogonal (transpose equals inverse).
- **Usage**: `matrix.is_orthogonal()`
- **Example**:
  
  m = Matrix([[1, 0], [0, -1]])

  print(m.is_orthogonal())  # Outputs: True
  

## Unitary Matrix Check (`is_unitary`)
- Checks if the matrix is unitary (conjugate transpose equals inverse).
- **Usage**: `matrix.is_unitary()`
- **Example**:
  
  m = Matrix([[1, 0], [0, 1]])

  print(m.is_unitary())  # Outputs: True
  

## Scalar Matrix Check (`is_scalar`)
- Checks if the matrix is a scalar matrix (diagonal with all diagonal elements equal).
- **Usage**: `matrix.is_scalar()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 2]])

  print(m.is_scalar())  # Outputs: True
  

## Singular Matrix Check (`is_singular`)
- Checks if the matrix is singular (determinant is 0).
- **Usage**: `matrix.is_singular()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.is_singular())  # Outputs: True
  

## Invertibility Check (`is_invertible`)
- Checks if the matrix is invertible (determinant is non-zero).
- **Usage**: `matrix.is_invertible()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.is_invertible())  # Outputs: True
  

## Identity Matrix Check (`is_identity`)
- Checks if the matrix is an identity matrix.
- **Usage**: `matrix.is_identity()`
- **Example**:
  
  m = Matrix([[1, 0], [0, 1]])

  print(m.is_identity())  # Outputs: True
  

## Nilpotent Matrix Check (`is_nilpotent`)
- Checks if the matrix is nilpotent (power of the matrix is the zero matrix).
- **Usage**: `matrix.is_nilpotent()`
- **Example**:
  
  m = Matrix([[0, 1], [0, 0]])

  print(m.is_nilpotent())  # Outputs: True
  

## Diagonalizable Matrix Check (`is_diagonalizable`)
- Checks if the matrix is diagonalizable.
- **Usage**: `matrix.is_diagonalizable()`
- **Example**:
  
  m = Matrix([[5, 4], [1, 2]])

  print(m.is_diagonalizable())  # Outputs: True
  

## LU Decomposition Check (`has_lu_decomposition`)
- Checks if the matrix can have an LU decomposition.
- **Usage**: `matrix.has_lu_decomposition()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.has_lu_decomposition())  # Outputs: True
  

## PLU Decomposition Check (`has_plu_decomposition`)
- Checks if the matrix can have a PLU decomposition.
- **Usage**: `matrix.has_plu_decomposition()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.has_plu_decomposition())  # Outputs: True
  

## Determinant (`determinant`)
- Computes the determinant of the matrix.
- **Usage**: `matrix.determinant()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.determinant())  # Outputs: -2
  

## Minor (`get_minor`)
- Computes the minor of a matrix element at a specific row and column.
- **Usage**: `matrix.get_minor(row, col)`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.get_minor(0, 0))  # Outputs: 4
  

## Size (`size`)
- Returns the size (rows, columns) of the matrix.
- **Usage**: `matrix.size()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.size())  # Outputs: (2, 2)
  


## Rank (`rank`)
- Computes the rank of the matrix (number of linearly independent rows or columns).
- **Usage**: `matrix.rank()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.rank())  # Outputs: 1
  

## Nullity (`nullity`)
- Computes the nullity of the matrix (dimension of the null space).
- **Usage**: `matrix.nullity()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.nullity())  # Outputs: 1
  

## Row-Reduced Echelon Form (`rref`)
- Computes the row-reduced echelon form of the matrix.
- **Usage**: `matrix.rref()`
- **Example**:
  
  m = Matrix([[1, 2, 1], [2, 4, 0]])

  print(m.rref())  # Outputs: Matrix([[1, 2, 1], [0, 0, -2]])
  

## Clean Matrix (`clean_matrix`)
- Rounds near-zero values in the matrix to exactly zero for numerical stability.
- **Usage**: `matrix.clean_matrix()`
- **Example**:
  
  m = Matrix([[1e-10, 1], [2, 3]])

  print(m.clean_matrix())  # Outputs: Matrix([[0, 1], [2, 3]])
  

## Create Identity Matrix (`create_identity_matrix`)
- Generates an identity matrix of size `n`.
- **Usage**: `Matrix.create_identity_matrix(n)`
- **Example**:
  
  id_matrix = Matrix.create_identity_matrix(3)

  print(id_matrix)  # Outputs: Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  

## Create Row Swap Matrix (`create_row_swap_matrix`)
- Generates a matrix that swaps two rows when multiplied by another matrix.
- **Usage**: `Matrix.create_row_swap_matrix(n, row1, row2)`
- **Example**:
  
  swap_matrix = Matrix.create_row_swap_matrix(3, 0, 1)

  print(swap_matrix)  # Outputs: Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
  

## Create Row Scale Matrix (`create_row_scale_matrix`)
- Generates a matrix that scales a row by a given factor.
- **Usage**: `Matrix.create_row_scale_matrix(n, row, factor)`
- **Example**:
  
  scale_matrix = Matrix.create_row_scale_matrix(3, 1, 2)

  print(scale_matrix)  # Outputs: Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
  

## Create Row Replacement Matrix (`create_row_replacement_matrix`)
- Generates a matrix that adds a scaled row to another row.
- **Usage**: `Matrix.create_row_replacement_matrix(n, source_row, target_row, factor)`
- **Example**:
  
  replacement_matrix = Matrix.create_row_replacement_matrix(3, 0, 1, 2)

  print(replacement_matrix)  # Outputs: Matrix([[1, 0, 0], [2, 1, 0], [0, 0, 1]])
  

## Linear Independence Check (`are_vectors_linearly_independent`)
- Checks if a set of vectors is linearly independent.
- **Usage**: `matrix.are_vectors_linearly_independent()`
- **Example**:
  
  m = Matrix([[1, 0], [0, 1]])

  print(m.are_vectors_linearly_independent())  # Outputs: True
  

## Dimension of Span (`dimension_of_span`)
- Computes the dimension of the span of the matrix rows.
- **Usage**: `matrix.dimension_of_span()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.dimension_of_span())  # Outputs: 1
  

## Basis for Span (`basis_for_span`)
- Finds a basis for the row span of the matrix.
- **Usage**: `matrix.basis_for_span()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.basis_for_span())  # Outputs: Matrix([[1, 2]])
  

## Rank Factorization (`rank_factorization`)
- Computes the rank factorization of the matrix.
- **Usage**: `matrix.rank_factorization()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.rank_factorization())  # Outputs: (Matrix([[1], [3]]), Matrix([[1, 2]]))
  

## LU Decomposition (`lu_decomposition`)
- Computes the LU decomposition of the matrix.
- **Usage**: `matrix.lu_decomposition()`
- **Example**:
  
  m = Matrix([[4, 3], [6, 3]])

  print(m.lu_decomposition())  # Outputs: (L, U)
  

## PLU Decomposition (`plu_decomposition`)
- Computes the PLU decomposition of the matrix.
- **Usage**: `matrix.plu_decomposition()`
- **Example**:
  
  m = Matrix([[4, 3], [6, 3]])

  print(m.plu_decomposition())  # Outputs: (P, L, U)
  

## Inverse (`inverse`)
- Computes the inverse of the matrix (if invertible).
- **Usage**: `matrix.inverse()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.inverse())  # Outputs: Matrix([[-2, 1], [1.5, -0.5]])
  

## Inverse by Adjoint (`inverse_by_adjoint`)
- Computes the inverse using the adjoint method.
- **Usage**: `matrix.inverse_by_adjoint()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.inverse_by_adjoint())  # Outputs: Matrix([[-2, 1], [1.5, -0.5]])
  



## Determinant via Cofactor (`determinant_cofactor`)
- Computes the determinant using cofactor expansion.
- **Usage**: `matrix.determinant_cofactor()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.determinant_cofactor())  # Outputs: -2
  

## Determinant via PLU (`determinant_plu`)
- Computes the determinant using PLU decomposition.
- **Usage**: `matrix.determinant_plu()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.determinant_plu())  # Outputs: -2
  

## Determinant via RREF (`determinant_rref`)
- Computes the determinant using row-reduced echelon form.
- **Usage**: `matrix.determinant_rref()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.determinant_rref())  # Outputs: -2
  

## QR Decomposition (`qr_decomposition`)
- Computes the QR decomposition of the matrix.
- **Usage**: `matrix.qr_decomposition()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.qr_decomposition())  # Outputs: (Q, R)
  

## Eigenvalues (`eigenvalues`)
- Computes the eigenvalues of the matrix.
- **Usage**: `matrix.eigenvalues()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.eigenvalues())  # Outputs: [2, 3]
  

## Eigenvectors (`eigenvectors`)
- Computes the eigenvectors of the matrix.
- **Usage**: `matrix.eigenvectors()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.eigenvectors())  # Outputs: [[1, 0], [0, 1]]
  

## Characteristic Polynomial (`characteristic_polynomial`)
- Computes the characteristic polynomial of the matrix.
- **Usage**: `matrix.characteristic_polynomial()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.characteristic_polynomial())  # Outputs: lambda^2 - 5*lambda + 6
  

## Minimal Polynomial (`minimal_polynomial`)
- Computes the minimal polynomial of the matrix.
- **Usage**: `matrix.minimal_polynomial()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.minimal_polynomial())  # Outputs: (lambda - 2)(lambda - 3)
  

## Generate Polynomial Coefficients (`generate_polynomial_coefficients`)
- Generates the coefficients of a polynomial for a given matrix.
- **Usage**: `matrix.generate_polynomial_coefficients()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.generate_polynomial_coefficients())  # Outputs: [1, -5, 6]
  

## Recursive Polynomial Generation (`recursive_generate`)
- Recursively generates a polynomial from the matrix.
- **Usage**: `matrix.recursive_generate()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.recursive_generate())  # Outputs: lambda^2 - 5*lambda + 6
  

## Evaluate Polynomial at Matrix (`evaluate_polynomial_at_matrix`)
- Evaluates a polynomial at the matrix.
- **Usage**: `matrix.evaluate_polynomial_at_matrix(polynomial)`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  p = lambda x: x**2 - 5*x + 6

  print(m.evaluate_polynomial_at_matrix(p))  # Outputs: Matrix([[0, 0], [0, 0]])
  

## Null Space (`null_space`)
- Computes the null space of the matrix.
- **Usage**: `matrix.null_space()`
- **Example**:
  
  m = Matrix([[1, 2], [2, 4]])

  print(m.null_space())  # Outputs: [[-2], [1]]
  

## Algebraic Multiplicity (`algebraic_multiplicity`)
- Computes the algebraic multiplicity of the eigenvalues.
- **Usage**: `matrix.algebraic_multiplicity()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 2]])

  print(m.algebraic_multiplicity())  # Outputs: {2: 2}
  

## Geometric Multiplicity (`geometric_multiplicity`)
- Computes the geometric multiplicity of the eigenvalues.
- **Usage**: `matrix.geometric_multiplicity()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 2]])

  print(m.geometric_multiplicity())  # Outputs: {2: 1}
  



## Eigen Basis (`eigen_basis`)
- Computes the eigenbasis of the matrix.
- **Usage**: `matrix.eigen_basis()`
- **Example**:
  
  m = Matrix([[2, 0], [0, 3]])

  print(m.eigen_basis())  # Outputs: [[1, 0], [0, 1]]
  

## Diagonalizable Check (`is_diagonalizable`)
- Checks if the matrix is diagonalizable.
- **Usage**: `matrix.is_diagonalizable()`
- **Example**:
  
  m = Matrix([[5, 4], [1, 2]])

  print(m.is_diagonalizable())  # Outputs: True
  

## Diagonalization Basis (`diagonalization_basis`)
- Computes the basis for diagonalization of the matrix.
- **Usage**: `matrix.diagonalization_basis()`
- **Example**:
  
  m = Matrix([[5, 4], [1, 2]])

  print(m.diagonalization_basis())  # Outputs: [[1, 0], [0, 1]]
  

## Pseudoinverse (`pseudoinverse`)
- Computes the Moore-Penrose pseudoinverse of the matrix.
- **Usage**: `matrix.pseudoinverse()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.pseudoinverse())  # Outputs: [[-2, 1], [1.5, -0.5]]
  

## Square Root of Matrix (`sqrt`)
- Computes the square root of the matrix, if it exists.
- **Usage**: `matrix.sqrt()`
- **Example**:
  
  m = Matrix([[4, 0], [0, 9]])

  print(m.sqrt())  # Outputs: Matrix([[2, 0], [0, 3]])
  

## Polar Decomposition (`polar_decomposition`)
- Computes the polar decomposition of the matrix.
- **Usage**: `matrix.polar_decomposition()`
- **Example**:
  
  m = Matrix([[1, 2], [3, 4]])

  print(m.polar_decomposition())  # Outputs: (U, P)
  



# Polynomial Roots: Aberth Method and Supporting Functions

## File Description
The `polynomialroots.py` file implements methods to find the roots of a polynomial using the Aberth method, along with supporting functions for polynomial evaluation, derivative computation, and initial root approximations.

## Aberth Method (`aberth_method`)
- Finds the roots of a polynomial using the Aberth-Ehrlich method, an iterative numerical approach.
- **Usage**: `aberth_method(coefficients, max_iterations, tolerance)`
- **Example**:
  
  coefficients = [1, -6, 11, -6]  # Represents x^3 - 6x^2 + 11x - 6

  roots = aberth_method(coefficients, max_iterations=100, tolerance=1e-6)

  print(roots)  # Outputs: [1.0, 2.0, 3.0] (approximate roots)
  

## Evaluate Polynomial (`evaluate_polynomial`)
- Evaluates the polynomial at a given point using the provided coefficients.
- **Usage**: `evaluate_polynomial(coefficients, x)`
- **Example**:
  
  coefficients = [1, -6, 11, -6]  # Represents x^3 - 6x^2 + 11x - 6

  result = evaluate_polynomial(coefficients, 2)

  print(result)  # Outputs: 0
  

## Evaluate Derivative (`evaluate_derivative`)
- Evaluates the derivative of the polynomial at a given point.
- **Usage**: `evaluate_derivative(coefficients, x)`
- **Example**:
  
  coefficients = [1, -6, 11, -6]  # Represents x^3 - 6x^2 + 11x - 6

  derivative_result = evaluate_derivative(coefficients, 2)

  print(derivative_result)  # Outputs: 3
  

## Initialize Roots (`initialize_roots`)
- Provides initial approximations for the roots of the polynomial.
- **Usage**: `initialize_roots(coefficients)`
- **Example**:
  
  coefficients = [1, -6, 11, -6]  # Represents x^3 - 6x^2 + 11x - 6

  initial_roots = initialize_roots(coefficients)

  print(initial_roots)  # Outputs: Complex initial approximations for roots
  



# System of Linear Equations: LinearSystem Class

## File Description
The `systemoflinear.py` file provides a `LinearSystem` class for representing and solving systems of linear equations. It includes methods for checking consistency, solving systems, identifying subspaces, and handling free variables in solutions. Advanced methods like PLU-based solving and least-squares solutions are also implemented.

## Initialization (`__init__`)
- Initializes a `LinearSystem` object with a coefficient matrix and a constant vector.
- **Usage**: `LinearSystem(matrix, constants)`
- **Example**:
  
  matrix = [[1, 2], [3, 4]]

  constants = [5, 6]

  system = LinearSystem(matrix, constants)

  print(system)  # Outputs: A representation of the system
  

## Consistency Check (`is_consistent`)
- Checks if the system of equations is consistent (i.e., has at least one solution).
- **Usage**: `system.is_consistent()`
- **Example**:
  
  matrix = [[1, 2], [2, 4]]

  constants = [5, 10]

  system = LinearSystem(matrix, constants)

  print(system.is_consistent())  # Outputs: True
  

## Solve System (`solve`)
- Solves the system of linear equations if it is consistent.
- **Usage**: `system.solve()`
- **Example**:
  
  matrix = [[1, 2], [3, 4]]

  constants = [5, 6]

  system = LinearSystem(matrix, constants)

  solution = system.solve()

  print(solution)  # Outputs: Solution vector, if it exists
  

## Subspace Check (`is_subspace`)
- Checks if a given vector is in the subspace defined by the system.
- **Usage**: `system.is_subspace(vector)`
- **Example**:
  
  matrix = [[1, 2], [3, 4]]

  constants = [0, 0]

  system = LinearSystem(matrix, constants)

  vector = [1, -0.5]

  print(system.is_subspace(vector))  # Outputs: True or False
  

## Solution Set with Free Variables (`solution_set_with_free_variables`)
- Finds the solution set for the system, expressing solutions in terms of free variables.
- **Usage**: `system.solution_set_with_free_variables()`
- **Example**:
  
  matrix = [[1, 2, 3], [4, 5, 6]]

  constants = [7, 8]
  system = LinearSystem(matrix, constants)

  solution_set = system.solution_set_with_free_variables()

  print(solution_set)  # Outputs: Solutions in parametric form
  

## Solve with PLU (`solve_with_plu`)
- Solves the system of equations using PLU decomposition.
- **Usage**: `system.solve_with_plu()`
- **Example**:
  
  matrix = [[1, 2], [3, 4]]

  constants = [5, 6]

  system = LinearSystem(matrix, constants)

  solution = system.solve_with_plu()

  print(solution)  # Outputs: Solution vector, if it exists
  

## Least-Square Solution (`least_square_solution`)
- Computes the least-squares solution for an inconsistent system.
- **Usage**: `system.least_square_solution()`
- **Example**:
  
  matrix = [[1, 1], [1, -1]]

  constants = [2, 0]

  system = LinearSystem(matrix, constants)

  least_square = system.least_square_solution()

  print(least_square)  # Outputs: Approximate solution vector
  


# Vector Class: Representation, Operations, and Properties

## Class Description
The `Vector` class provides a robust representation of mathematical vectors and includes essential methods for vector arithmetic, validation, and analysis. It supports operations like addition, subtraction, negation, and scalar multiplication. The class also includes functions for computing the vector length, inner product, and determining orthogonality. Overloaded operators allow seamless integration with Python's arithmetic and comparison expressions.

## Initialization (`__init__`)
- Initializes a `Vector` object with a list of numerical components.
- **Usage**: `Vector(components)`
- **Example**:
  
  v = Vector([1, 2, 3])

  print(v)  # Outputs: Vector([1, 2, 3])
  

## Validate Same Type and Length (`_validate_same_type_and_length`)
- Ensures two vectors are of the same type and length before performing operations.
- **Usage**: `_validate_same_type_and_length(vector1, vector2)`
- **Example**:
  
  v1 = Vector([1, 2, 3])

  v2 = Vector([4, 5, 6])

  v1._validate_same_type_and_length(v2)  # No error
  

## String Representation (`__str__`)
- Returns a human-readable string representation of the vector.
- **Usage**: `str(vector)`
- **Example**:
  
  v = Vector([1, 2, 3])

  print(str(v))  # Outputs: "[1, 2, 3]"
  

## Developer-Friendly Representation (`__repr__`)
- Returns a detailed representation of the vector for debugging.
- **Usage**: `repr(vector)`
- **Example**:
  
  v = Vector([1, 2, 3])

  print(repr(v))  # Outputs: "Vector([1, 2, 3])"
  

## Addition (`__add__`)
- Adds two vectors component-wise.
- **Usage**: `vector1 + vector2`
- **Example**:
  
  v1 = Vector([1, 2, 3])

  v2 = Vector([4, 5, 6])

  print(v1 + v2)  # Outputs: Vector([5, 7, 9])
  

## Negation (`__neg__`)
- Negates the components of a vector.
- **Usage**: `-vector`
- **Example**:
  
  v = Vector([1, 2, 3])

  print(-v)  # Outputs: Vector([-1, -2, -3])
  

## Subtraction (`__sub__`)
- Subtracts one vector from another component-wise.
- **Usage**: `vector1 - vector2`
- **Example**:
  
  v1 = Vector([4, 5, 6])

  v2 = Vector([1, 2, 3])

  print(v1 - v2)  # Outputs: Vector([3, 3, 3])
  

## Reverse Subtraction (`__rsub__`)
- Handles reversed subtraction of vectors.
- **Usage**: `vector1 - vector2` (reversed)
- **Example**:
  
  v1 = Vector([1, 2, 3])

  v2 = Vector([4, 5, 6])

  print(v2 - v1)  # Outputs: Vector([3, 3, 3])
  

## Scalar Multiplication (`__mul__`)
- Multiplies a vector by a scalar.
- **Usage**: `vector * scalar`
- **Example**:
  
  v = Vector([1, 2, 3])

  print(v * 2)  # Outputs: Vector([2, 4, 6])
  

## Division by Scalar (`__truediv__`)
- Divides a vector by a scalar.
- **Usage**: `vector / scalar`
- **Example**:
  
  v = Vector([2, 4, 6])

  print(v / 2)  # Outputs: Vector([1, 2, 3])
  

## Equality Check (`__eq__`)
- Checks if two vectors are equal component-wise.
- **Usage**: `vector1 == vector2`
- **Example**:
  
  v1 = Vector([1, 2, 3])
  
  v2 = Vector([1, 2, 3])

  print(v1 == v2)  # Outputs: True
  

## Inequality Check (`__ne__`)
- Checks if two vectors are not equal component-wise.
- **Usage**: `vector1 != vector2`
- **Example**:
  
  v1 = Vector([1, 2, 3])

  v2 = Vector([4, 5, 6])

  print(v1 != v2)  # Outputs: True
  

## Absolute Value (`__abs__`)
- Computes the magnitude (length) of the vector.
- **Usage**: `abs(vector)`
- **Example**:
  
  v = Vector([3, 4])

  print(abs(v))  # Outputs: 5.0
  

## Vector Length (`vector_length`)
- Computes the magnitude (length) of the vector.
- **Usage**: `vector.vector_length()`
- **Example**:
  
  v = Vector([3, 4])

  print(v.vector_length())  # Outputs: 5.0
  

## Inner Product (`inner_product`)
- Computes the dot product of two vectors.
- **Usage**: `vector1.inner_product(vector2)`
- **Example**:
  
  v1 = Vector([1, 2, 3])

  v2 = Vector([4, 5, 6])

  print(v1.inner_product(v2))  # Outputs: 32
  

## Orthogonality Check (`is_orthogonal`)
- Checks if two vectors are orthogonal (dot product is zero).
- **Usage**: `vector1.is_orthogonal(vector2)`
- **Example**:
  
  v1 = Vector([1, 0])

  v2 = Vector([0, 1])

  print(v1.is_orthogonal(v2))  # Outputs: True
  


# Vector Space: VectorSpace Class

## File Description
The `vectorspace.py` file provides a `VectorSpace` class that represents vector spaces and implements methods for analyzing spans, transformations, and basis-related operations. It includes functionality for checking span inclusion, basis transformations, and orthogonalization using the Gram-Schmidt process.

## Initialization (`__init__`)
- Initializes a `VectorSpace` object with a set of basis vectors.
- **Usage**: `VectorSpace(basis_vectors)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  print(space)  # Outputs: A representation of the vector space
  

## Check Span Inclusion (`is_in_span`)
- Checks if a vector is in the span of the vector space.
- **Usage**: `space.is_in_span(vector)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  vector = [1, 1]

  print(space.is_in_span(vector))  # Outputs: True
  

## Representation in Span (`representation_in_span`)
- Finds the representation of a vector as a linear combination of the basis vectors.
- **Usage**: `space.representation_in_span(vector)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  vector = [1, 2]

  representation = space.representation_in_span(vector)

  print(representation)  # Outputs: [1, 2]
  

## Subspace Span Comparison (`spans_same_subspace`)
- Checks if another set of vectors spans the same subspace as the current vector space.
- **Usage**: `space.spans_same_subspace(other_vectors)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  other_vectors = [[1, 1], [1, -1]]

  print(space.spans_same_subspace(other_vectors))  # Outputs: True
  

## Coordinates in Basis (`coordinates_in_basis`)
- Computes the coordinates of a vector in the given basis.
- **Usage**: `space.coordinates_in_basis(vector)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  vector = [2, 3]

  coordinates = space.coordinates_in_basis(vector)

  print(coordinates)  # Outputs: [2, 3]
  

## Vector from Coordinates (`vector_from_coordinates`)
- Reconstructs a vector from its coordinates in the given basis.
- **Usage**: `space.vector_from_coordinates(coordinates)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  space = VectorSpace(basis_vectors)

  coordinates = [2, 3]

  vector = space.vector_from_coordinates(coordinates)

  print(vector)  # Outputs: [2, 3]
  

## Change of Basis (`change_of_basis`)
- Transforms a vector from the current basis to a new basis.
- **Usage**: `space.change_of_basis(vector, new_basis)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  new_basis = [[1, 1], [1, -1]]

  space = VectorSpace(basis_vectors)

  vector = [2, 3]

  transformed_vector = space.change_of_basis(vector, new_basis)

  print(transformed_vector)  # Outputs: Transformed coordinates
  

## Coordinates in New Basis (`coordinates_in_new_basis`)
- Computes the coordinates of a vector in a new basis.
- **Usage**: `space.coordinates_in_new_basis(vector, new_basis)`
- **Example**:
  
  basis_vectors = [[1, 0], [0, 1]]

  new_basis = [[1, 1], [1, -1]]

  space = VectorSpace(basis_vectors)

  vector = [2, 3]

  new_coordinates = space.coordinates_in_new_basis(vector, new_basis)

  print(new_coordinates)  # Outputs: Coordinates in the new basis
  

## Gram-Schmidt Process (`gram_schmidt`)
- Performs the Gram-Schmidt orthogonalization on the basis vectors.
- **Usage**: `space.gram_schmidt()`
- **Example**:
  
  basis_vectors = [[1, 1], [1, -1]]

  space = VectorSpace(basis_vectors)

  orthogonal_basis = space.gram_schmidt()
  
  print(orthogonal_basis)  # Outputs: Orthogonalized basis vectors
  



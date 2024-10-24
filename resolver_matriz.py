import numpy as np
from colorama import Fore, Style, init

# Inicializar colorama
init(autoreset=True)

# Definimos la matriz A
A = np.array([
    [2 + 1j, 3, 1 - 1j, 4],
    [1, 4 + 2j, 2, 3],
    [3, 1 - 1j, 4, 2],
    [0 + 2j, 2, 4, 3]
], dtype=np.complex128)

def round_matrix(matrix, decimals=5):
    return np.round(matrix, decimals=decimals)

def get_color_for_value(value):
    """Determina el color apropiado para un valor dado"""
    if abs(value) < 1e-10:  # Casi cero
        return Fore.BLUE
    elif abs(value.imag) < 1e-10:  # Real
        if abs(value.real - 1) < 1e-10:  # Cercano a 1
            return Fore.GREEN
        elif abs(value.real) < 1e-10:  # Cercano a 0
            return Fore.BLUE
        else:
            return Fore.YELLOW
    else:  # Complejo
        return Fore.MAGENTA

def format_complex_number(value, decimals=5):
    """Formatea un número complejo para su visualización"""
    if abs(value) < 1e-10:
        return f"{0:10.{decimals}f}"
    elif abs(value.imag) < 1e-10:
        return f"{value.real:10.{decimals}f}"
    else:
        return f"{value:10.{decimals}f}"

def print_matrix_colored(matrix, message, show_operations=None):
    """Imprime la matriz con colores y formato mejorado"""
    print(f"\n{Fore.CYAN}{message}{Style.RESET_ALL}")
    print("=" * 60)
    
    if show_operations:
        print(f"{Fore.YELLOW}Operaciones realizadas:{Style.RESET_ALL}")
        for op in show_operations:
            print(f"→ {op}")
        print("-" * 60)
    
    for i, row in enumerate(matrix):
        print("│ ", end="")
        for j, value in enumerate(row):
            color = get_color_for_value(value)
            formatted_value = format_complex_number(value)
            print(f"{color}{formatted_value}{Style.RESET_ALL}", end="  ")
        
        # Mostrar indicadores de fila
        print("│", end="")
        if i == len(matrix) // 2:
            print(f" {Fore.CYAN}(Matriz){Style.RESET_ALL}")
        else:
            print()
    
    print("=" * 60)

def print_step_info(step_number, total_steps, operation):
    """Imprime información sobre el progreso del proceso"""
    progress = (step_number / total_steps) * 100
    print(f"\n{Fore.YELLOW}Paso {step_number}/{total_steps} ({progress:.1f}%){Style.RESET_ALL}")
    print(f"{Fore.CYAN}Operación: {operation}{Style.RESET_ALL}")

def gauss_jordan_elimination(A):
    A = A.copy()
    n = len(A)
    total_steps = n * 2  # Pasos totales (eliminación hacia adelante y hacia atrás)
    current_step = 0
    
    print_matrix_colored(round_matrix(A), "Matriz inicial A:")
    
    # Fase de eliminación hacia adelante (Gauss)
    for i in range(n):
        current_step += 1
        
        # Normalización de la fila actual
        pivot = A[i, i]
        old_row = A[i].copy()
        A[i] = A[i] / pivot
        A = round_matrix(A)
        
        operations = [
            f"Normalización de fila {i+1}: dividir por {format_complex_number(pivot)}",
            f"Pivote en posición ({i+1},{i+1})"
        ]
        
        print_step_info(current_step, total_steps, "Normalización de fila")
        print_matrix_colored(A, f"Después de normalizar fila {i+1}", operations)
        
        # Hacer ceros debajo del pivote
        operations = []
        for j in range(i + 1, n):
            factor = A[j, i]
            if abs(factor) > 1e-10:
                A[j] = A[j] - factor * A[i]
                operations.append(f"F{j+1} = F{j+1} - ({format_complex_number(factor)}) × F{i+1}")
        
        A = round_matrix(A)
        if operations:
            print_step_info(current_step, total_steps, "Eliminación hacia abajo")
            print_matrix_colored(A, f"Después de hacer ceros debajo del pivote en columna {i+1}", operations)
    
    # Fase de eliminación hacia atrás (Jordan)
    for i in range(n-1, -1, -1):
        current_step += 1
        operations = []
        
        # Hacer ceros encima del pivote
        for j in range(i-1, -1, -1):
            factor = A[j, i]
            if abs(factor) > 1e-10:
                A[j] = A[j] - factor * A[i]
                operations.append(f"F{j+1} = F{j+1} - ({format_complex_number(factor)}) × F{i+1}")
        
        A = round_matrix(A)
        if operations:
            print_step_info(current_step, total_steps, "Eliminación hacia arriba")
            print_matrix_colored(A, f"Después de hacer ceros encima del pivote en columna {i+1}", operations)
    
    return A

# Ejecutar el método de Gauss-Jordan
print(f"{Fore.CYAN}Iniciando método de Gauss-Jordan...{Style.RESET_ALL}")
print("-" * 60)
resultado = gauss_jordan_elimination(A)

# Imprimir el resultado final
print_matrix_colored(resultado, "RESULTADO FINAL - Matriz en forma escalonada reducida:")

# Verificar que el resultado es correcto
identidad = np.eye(len(A))
error = np.abs(resultado - identidad).max()
print(f"\n{Fore.GREEN}Verificación de resultados:{Style.RESET_ALL}")
print(f"Error máximo respecto a la matriz identidad: {error:.2e}")

if error < 1e-10:
    print(f"{Fore.GREEN}✓ La matriz se ha reducido correctamente a la forma escalonada reducida{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}⚠ La matriz puede tener algunas desviaciones numéricas{Style.RESET_ALL}")
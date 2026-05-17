#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <mpi.h>
#include <cstring>
#include <algorithm>
using namespace std;
using namespace chrono;

// Структура для хранения информации о разбиении матрицы
struct MatrixPartition {
    int* rows_per_proc;
    int* displs;
    MatrixPartition(int n, int size) {
        rows_per_proc = new int[size];
        displs = new int[size];
        int base_rows = n / size;
        int remainder = n % size;
        for (int i = 0; i < size; i++) {
            rows_per_proc[i] = base_rows + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + rows_per_proc[i-1];
        }
    }
    ~MatrixPartition() {
        delete[] rows_per_proc;
        delete[] displs;
    }
    int getLocalRows(int rank) const {
        return rows_per_proc[rank];
    }
    int getDispl(int rank) const {
        return displs[rank];
    }
};
// Выделение памяти в общей памяти MPI
double* allocateSharedMatrix(int n, int rank, int size, MPI_Win& win, int& local_rows) {
    MatrixPartition partition(n, size);
    local_rows = partition.getLocalRows(rank);
    MPI_Aint local_size = local_rows * n * sizeof(double);
    double* local_matrix = nullptr;
    // Используем MPI_Win_allocate_shared для выделения общей памяти
    MPI_Win_allocate_shared(local_size, sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &local_matrix, &win);
    // Обнуляем локальную часть
    if (local_matrix != nullptr) {
        memset(local_matrix, 0, local_size);
    }
    return local_matrix;
}

// Инициализация матрицы случайными числами в общей памяти
void initMatrixShared(double* local_matrix, int n, int rank, int size,  double start, double end, unsigned int seed) {
    MatrixPartition partition(n, size);
    int local_rows = partition.getLocalRows(rank);
    random_device rd;
    mt19937 gen(rd() + seed + rank);
    uniform_real_distribution<double> dist(start, end);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_matrix[i * n + j] = dist(gen);
        }
    }
    // Синхронизация после инициализации
    MPI_Barrier(MPI_COMM_WORLD);
}

// MPI умножение с использованием общей памяти
void multiplyMPIShared(double* local_A, double* local_B, double* local_C,  int n, int rank, int size, MPI_Win win_A, MPI_Win win_B) {
    MatrixPartition partition(n, size);
    int local_rows = partition.getLocalRows(rank);
    // Получаем указатель на полную матрицу B из общей памяти (только для чтения)
    double* full_B = nullptr;
    MPI_Aint b_size = 0;
    int disp_unit = 0;
    MPI_Win_shared_query(win_B, 0, &b_size, &disp_unit, &full_B);
    // Блокируем окно для чтения
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_B);
    // Каждый процесс умножает свои строки A на полную матрицу B
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * full_B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
    MPI_Win_unlock(0, win_B);
    MPI_Barrier(MPI_COMM_WORLD);
}
// Функция для тестирования одного размера матрицы
void testMatrixSize(int test_n, int rank, int size, double serial_time, bool print_header = false) {
    if (rank == 0 && print_header) {
        cout << left << setw(10) << "Размер"  << " | " << setw(20) << "Последовательно (мс)" << " | " << setw(20) << "MPI общая память (мс)"  << " | " << setw(12) << "Ускорение" << endl;
        cout << string(75, '-') << endl;
    }
    MPI_Win win_A, win_B;
    double *local_A = nullptr, *local_B = nullptr, *local_C = nullptr;
    int local_rows_A = 0, local_rows_B = 0;
    // Выделяем общую память для матриц A и B
    local_A = allocateSharedMatrix(test_n, rank, size, win_A, local_rows_A);
    local_B = allocateSharedMatrix(test_n, rank, size, win_B, local_rows_B);
    if (local_A == nullptr || local_B == nullptr) {
        MPI_Win_free(&win_A);
        MPI_Win_free(&win_B);
        return;
    }
    // Выделяем память для локальной матрицы C (результат)
    local_C = new double[local_rows_A * test_n]();
    // Инициализируем матрицы
    initMatrixShared(local_A, test_n, rank, size, -10.0, 10.0, 0);
    initMatrixShared(local_B, test_n, rank, size, -10.0, 10.0, 100);
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI умножение
    double mpi_start = MPI_Wtime();
    multiplyMPIShared(local_A, local_B, local_C, test_n, rank, size, win_A, win_B);
    double mpi_end = MPI_Wtime();
    double mpi_time = (mpi_end - mpi_start) * 1000.0;
    // Вывод результатов
    if (rank == 0) {
        double speedup = serial_time / mpi_time;
        cout << left << setw(10) << test_n << " | " << setw(20) << fixed << setprecision(2) << serial_time
             << " | " << setw(20) << fixed << setprecision(2) << mpi_time<< " | " << setw(12) << fixed << setprecision(2) << speedup << "x" << endl;
    }
    // Очистка
    delete[] local_C;
    MPI_Win_free(&win_A);
    MPI_Win_free(&win_B);
    MPI_Barrier(MPI_COMM_WORLD);
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Вектор размеров матриц
    vector<int> test_sizes = {500, 1000, 2000, 3000, 4000};
    vector<double> serial_times = {478.00, 4789.00, 50262.00, 162379.00, 611926.00};
    if (rank == 0) {
        cout << "MPI УМНОЖЕНИЕ МАТРИЦ" << endl;
        cout << "Количество MPI процессов: " << size << endl;
        cout << endl;
        cout << "ТЕСТИРОВАНИЕ ДЛЯ РАЗНЫХ РАЗМЕРОВ МАТРИЦ" << endl;
        cout << endl;
    }
    for (size_t i = 0; i < test_sizes.size(); i++) {
        // Пропускаем, если время не задано (0)
        if (serial_times[i] <= 0.0) {
            if (rank == 0) {
                cout << left << setw(10) << test_sizes[i]  << " | " << setw(20) << "Нет данных" << " | " << setw(20) << "Пропущено" << " | " << setw(12) << "-" << endl;
            }
            continue;
        }
        testMatrixSize(test_sizes[i], rank, size, serial_times[i], i == 0);
    }
    MPI_Finalize();
    return 0;
}

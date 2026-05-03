#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <immintrin.h>  
using namespace std;
using namespace chrono;

double* RandomNumbers(int n, int start, int end) {
    double* Vec = new double[n * n];
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(start, end);
    for (int i = 0; i < n * n; i++) {
        Vec[i] = dist(gen);
    }
    return Vec;
}
double* DGEMM_BLAS(double* a, double* b, int n) {
    double* c = new double[n * n]();
    for (int i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    return c;
}
double* DGEMM_opt_1(double* a, double* b, int n) {
    double* c = new double[n * n]();
    for (int i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = a[i * n + k];
            double* b_row = &b[k * n];
            double* c_row = &c[i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return c;
}
double* DGEMM_opt_2(double* a, double* b, int n, int block) {
    double* c = new double[n * n]();
    for (int i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }
    for (int i = 0; i < n; i += block) {
        for (int j = 0; j < n; j += block) {
            for (int k = 0; k < n; k += block) {
                int maxI = min(i + block, n);
                int maxJ = min(j + block, n);
                int maxK = min(k + block, n);
                for (int ii = i; ii < maxI; ii++) {
                    for (int kk = k; kk < maxK; kk++) {
                        double aik = a[ii * n + kk];
                        double* b_row = &b[kk * n];
                        double* c_row = &c[ii * n];
                        for (int jj = j; jj < maxJ; jj++) {
                            c_row[jj] += aik * b_row[jj];
                        }
                    }
                }
            }
        }
    }
    return c;
}
double* DGEMM_opt_3(double* a, double* b, int n) {
    const int alignment = 32; // 32 байта для AVX
    double* c = (double*)aligned_alloc(alignment, n * n * sizeof(double));
    if (!c) {
        c = new double[n * n]();
    }
    memset(c, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = a[i * n + k];
            double* b_row = &b[k * n];
            double* c_row = &c[i * n];
            int j = 0;
            // Основной AVX цикл (по 4 элемента)
            for (; j <= n - 4; j += 4) {
                __m256d vec_b = _mm256_loadu_pd(&b_row[j]);      
                __m256d vec_c = _mm256_load_pd(&c_row[j]);       
                __m256d vec_aik = _mm256_set1_pd(aik);
                // c = c + aik * b
                __m256d product = _mm256_mul_pd(vec_aik, vec_b);
                vec_c = _mm256_add_pd(vec_c, product);
                _mm256_store_pd(&c_row[j], vec_c);
            }
            for (; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return c;
}
bool ValidateResult(double* c1, double* c2, int n) {
    const double epsilon = 1e-6;
    for (int i = 0; i < n * n; i++) {
        if (fabs(c1[i] - c2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}
// Измерение времени для opt_2 с блоком
double MeasureTimeBlock(int size, double* (*multiply_func)(double*, double*, int, int), int block_size, int iterations) {
    double total_time = 0.0;
    for (int t = 0; t < iterations; t++) {
        double* A = RandomNumbers(size, 1, 10);
        double* B = RandomNumbers(size, 1, 10);
        auto start = high_resolution_clock::now();
        double* C = multiply_func(A, B, size, block_size);
        auto end = high_resolution_clock::now();
        duration<double> time = end - start;
        total_time += time.count();
        delete[] A;
        delete[] B;
        delete[] C;
    }
    return total_time / iterations;
}
// Общее измерение для функций 
double MeasureTime(int size, double* (*multiply_func)(double*, double*, int), int iterations) {
    double total_time = 0.0;
    for (int t = 0; t < iterations; t++) {
        double* A = RandomNumbers(size, 1, 10);
        double* B = RandomNumbers(size, 1, 10);
        auto start = high_resolution_clock::now();
        double* C = multiply_func(A, B, size);
        auto end = high_resolution_clock::now();
        duration<double> time = end - start;
        total_time += time.count();
        delete[] A;
        delete[] B;
        if (multiply_func == DGEMM_opt_3) {
            free(C);
        } else {
            delete[] C;
        }
    }
    return total_time / iterations;
}
// Вычисление производительности (GFLOPS)
double CalculateGflops(int size, double time_seconds) {
    double operations = 2.0 * size * size * size;
    return operations / (time_seconds * 1e9);
}
// Пункт 1: Тестирование для заданного размера (сравнение всех реализаций)
void TestSingleSize(int size, int blocksize) {
    cout << "Умножение матриц размера " << size << " x " << size << endl;
    cout << "Размер блока для opt_2: " << blocksize << endl;

    cout << "\nПроверка корректности умножения" << endl;
    double* A_test = RandomNumbers(size, 1, 10);
    double* B_test = RandomNumbers(size, 1, 10);
    double* C_st = DGEMM_BLAS(A_test, B_test, size);
    double* C_opt1 = DGEMM_opt_1(A_test, B_test, size);
    double* C_opt2 = DGEMM_opt_2(A_test, B_test, size, blocksize);
    double* C_opt3 = DGEMM_opt_3(A_test, B_test, size);
    bool valid1 = ValidateResult(C_st, C_opt1, size);
    bool valid2 = ValidateResult(C_st, C_opt2, size);
    bool valid3 = ValidateResult(C_st, C_opt3, size);
    cout << "DGEMM_opt_1 (построчный): " << (valid1 ? "Все ок" : "НЕТ!") << endl;
    cout << "DGEMM_opt_2 (блочный):   " << (valid2 ? "Все ок" : "НЕТ!") << endl;
    cout << "DGEMM_opt_3 (векторный): " << (valid3 ? "Все ок" : "НЕТ!") << endl;
    delete[] A_test;
    delete[] B_test;
    delete[] C_st;
    delete[] C_opt1;
    delete[] C_opt2;
    free(C_opt3);  
    cout << endl;

    // Тестирование производительности
    cout << "\nСравнение производительности для введенных данных" << endl;
    cout << "|       Метод      | Время (сек) |  GFLOPS  | Ускорение |" << endl;
    cout << "|_______________________________________________________|" << endl;
    double time_standard = MeasureTime(size, DGEMM_BLAS, 1);
    double gflops_standard = CalculateGflops(size, time_standard);
    cout << "|    Стандартный   | "  << fixed << setprecision(4) << setw(10) << time_standard << " | " << setprecision(2) << setw(7) << gflops_standard << " |   1.00x   |" << endl;
    double time_opt1 = MeasureTime(size, DGEMM_opt_1, 1);
    double gflops_opt1 = CalculateGflops(size, time_opt1);
    double speedup1 = time_standard / time_opt1;
    cout << "|       Opt1       | " << fixed << setprecision(4) << setw(10) << time_opt1 << " | "  << setprecision(2) << setw(7) << gflops_opt1 << " | "  << setprecision(2) << setw(8) << speedup1 << "x |" << endl;
    
    double time_opt2 = MeasureTimeBlock(size, DGEMM_opt_2, blocksize, 1);
    double gflops_opt2 = CalculateGflops(size, time_opt2);
    double speedup2 = time_standard / time_opt2;
    cout << "|  Opt 2 (bs=" << setw(2) << blocksize << ")   | " << fixed << setprecision(4) << setw(10) << time_opt2 << " | " << setprecision(2) << setw(7) << gflops_opt2 << " | " << setprecision(2) << setw(8) << speedup2 << "x |" << endl;
    
    double time_opt3 = MeasureTime(size, DGEMM_opt_3, 1);
    double gflops_opt3 = CalculateGflops(size, time_opt3);
    double speedup3 = time_standard / time_opt3;
    cout << "|      Opt 3       | " << fixed << setprecision(4) << setw(10) << time_opt3 << " | "<< setprecision(2) << setw(7) << gflops_opt3 << " | " << setprecision(2) << setw(8) << speedup3 << "x |" << endl;
    
    cout <<  "\nЛучший результат: ";
    double best_speedup = max({speedup1, speedup2, speedup3});
    if (best_speedup == speedup1) cout << "opt 1 (построчный)";
    else if (best_speedup == speedup2) cout << "opt 2 (блочный)";
    else cout << "opt 3 (векторный)";
    cout << " с ускорением " << fixed << setprecision(2) << best_speedup << "x" << endl;
}

// Пункт 2 + 7: Серия испытаний для разных размеров
void RunBenchmark() {
    vector<int> sizes = {500, 1000, 2000};
    
    // Таблица для стандартного метода
    cout << "\nDGEMM_BLAS" << endl;
    cout << "|  N   |   Время (сек)  |   GFLOPS   |" << endl;
    cout << "|______|________________|____________|" << endl;
    for (int size : sizes) {
        double time_standard = MeasureTime(size, DGEMM_BLAS, 1);
        double gflops_standard = CalculateGflops(size, time_standard);
        cout << "| " << setw(4) << size << " | "  << fixed << setprecision(4) << setw(12) << time_standard << " | " << setprecision(2) << setw(10) << gflops_standard << " |" << endl;
    }
    
    // Таблица для opt1 (построчный)
    cout << "\nDGEMM_opt_1" << endl;
    cout << "|  N   |   Время (сек)  |   GFLOPS   |" << endl;
    cout << "|______|________________|____________|" << endl;
    for (int size : sizes) {
        double time_opt1 = MeasureTime(size, DGEMM_opt_1, 1);
        double gflops_opt1 = CalculateGflops(size, time_opt1);
        cout << "| " << setw(4) << size << " | " << fixed << setprecision(4) << setw(12) << time_opt1 << " | "<< setprecision(2) << setw(10) << gflops_opt1 << " |" << endl;
    }
    
    // Таблица для opt3 (векторный AVX)
    cout << "\nDGEMM_opt_3" << endl;
    cout << "|  N   |   Время (сек)  |   GFLOPS   |" << endl;
    cout << "|______|________________|____________|" << endl;
    for (int size : sizes) {
        double time_opt3 = MeasureTime(size, DGEMM_opt_3, 1);
        double gflops_opt3 = CalculateGflops(size, time_opt3);
        cout << "| " << setw(4) << size << " | "  << fixed << setprecision(4) << setw(12) << time_opt3 << " | "  << setprecision(2) << setw(10) << gflops_opt3 << " |" << endl;
    }
}
// Пункт 7 (поиск оптимального блока)
void FindOptimalBlockSize() {
    vector<int> sizes = {500, 1000, 2000};
    vector<int> block_sizes = {2, 4, 8, 16, 32, 64};
    cout << "\nDGEMM_opt_2 - поиск оптимального блока" << endl;
    for (int size : sizes) {
        cout << "\nМатрица " << size << "x" << size << endl;
        cout << " Размер блока |   Время (сек)  |   GFLOPS   | Ускорение" << endl;
        cout << "______________|________________|____________|__________" << endl;
        double time_standard = MeasureTime(size, DGEMM_BLAS, 1);
        double best_time = time_standard;
        int best_block = 1;
        for (int bs : block_sizes) {
            if (bs > size) continue;
            double time_block = MeasureTimeBlock(size, DGEMM_opt_2, bs, 1);
            double gflops = CalculateGflops(size, time_block);
            double speedup = time_standard / time_block;
            cout << "     " << setw(3) << bs << "      |    " << fixed << setprecision(4) << setw(10) << time_block << "  |   " << setprecision(2) << setw(7) << gflops << "   |   " << setprecision(2) << setw(7) << speedup << " x" << endl;
            if (time_block < best_time) {
                best_time = time_block;
                best_block = bs;
            }
        }
        cout << "\nОптимальный размер блока: " << best_block << endl;
        cout << "Максимальное ускорение: " << fixed << setprecision(2) << (time_standard / best_time) << "x" << endl;
    }
}
void EstimateMaxSize() {
    cout << "\nОценка предельных размеров матриц (Только теория, от практики завис компьютер)" << endl;
    // Теоретический расчет
    const double total_ram_gb = 16.0;                // 16 Gb RAM
    const double available_gb = total_ram_gb * 0.8;  // 80% от RAM = 12.8 Gb
    const size_t bytes_per_element = sizeof(double); // 8 байт
    const int num_matrices = 3;                      // Матрицы A, B, C
    // Расчет максимального количества элементов
    size_t max_elements_total = static_cast<size_t>((available_gb * 1024 * 1024 * 1024) / bytes_per_element);
    size_t max_elements_per_matrix = max_elements_total / num_matrices;
    size_t max_size = static_cast<size_t>(sqrt(max_elements_per_matrix));
    cout << "\nИсходные данные:" << endl;
    cout << "  Объем RAM: " << total_ram_gb << " GB" << endl;
    cout << "  Используем 80%: " << available_gb << " GB" << endl;
    cout << "  Размер double: " << bytes_per_element << " байт" << endl;
    cout << "  Количество матриц: " << num_matrices << " (A, B, C)" << endl;
    cout << "\nРасчет:" << endl;
    cout << "  Всего элементов double в " << available_gb << " GB: " << max_elements_total << " ("  << fixed << setprecision(2) << max_elements_total / 1e9 << " млрд)" << endl;
    cout << "  Элементов на 1 матрицу: " << max_elements_per_matrix << " ("  << max_elements_per_matrix / 1e6 << " млн)" << endl;
    cout << "  Размер одной матрицы в памяти: " << (max_elements_per_matrix * bytes_per_element) / (1024.0 * 1024 * 1024) << " GB" << endl;
    cout << "\nРЕЗУЛЬТАТ:" << endl;
    cout << "  Максимальный размер квадратной матрицы: " << max_size << " x " << max_size << endl;
    cout << "\nФормула для расчета:" << endl;
    cout << "  N_max = sqrt(0.8 * RAM_GB * 1024^3 / (3 * sizeof(double)))" << endl;
    cout << "  N_max = sqrt(" << available_gb << " * 1073741824 / (3 * 8)) = " << max_size << endl;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    if (argc < 2 || argc > 3) {
        cout << "\nИспользование: " << argv[0] << " <размер матрицы> [размер блока]" << endl;
        cout << "Пример: " << argv[0] << " 1000 32" << endl;
        return -1;
    }
    int size = atoi(argv[1]);
    if (size <= 0) {
        cout << "Неверный размер матрицы. Размер должен быть положительным целым числом." << endl;
        return -1;
    }
    int block_size = 32; 
    if (argc == 3) {
        block_size = atoi(argv[2]);
        if (block_size <= 0) {
            cout << "Неверный размер блока. Используется значение по умолчанию 32." << endl;
            block_size = 32;
        }
    }
    // Пункт 3
    EstimateMaxSize();

    // Пункт 1, 4, 5, 6, 8
    //TestSingleSize(size, block_size);

    // Пункт 2, 7
    //RunBenchmark();
    //FindOptimalBlockSize();

    return 0;
}

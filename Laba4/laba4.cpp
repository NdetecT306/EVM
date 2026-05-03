#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <pthread.h>
#include <omp.h>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
using namespace std;
using namespace chrono;
using namespace tbb;

// Структуры для потоков
struct InitThreadData {
    double* matrix;
    int start_idx;
    int end_idx;
    double start_val;
    double end_val;
    unsigned int seed;
};
struct MultiplyThreadData {
    double* a;
    double* b;
    double* c;
    int n;
    int start_row;
    int end_row;
};

// OPENCL KERNEL (ПРОГРАММА ДЛЯ GPU) 
const char* opencl_kernel_source = R"(
__kernel void matrixMulKernel(
    __global const double* a,
    __global const double* b,
    __global double* c,
    int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < n && j < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}
__kernel void matrixMulBlockKernel(
    __global const double* a,
    __global const double* b,
    __global double* c,
    int n,
    __local double* shared_a,
    __local double* shared_b)
{
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    double sum = 0.0;
    for (int k = 0; k < n; k += 32) {
        if (row < n && k + tx < n)
            shared_a[ty * 32 + tx] = a[row * n + k + tx];
        else
            shared_a[ty * 32 + tx] = 0.0;
        if (col < n && k + ty < n)
            shared_b[ty * 32 + tx] = b[(k + ty) * n + col];
        else
            shared_b[ty * 32 + tx] = 0.0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < 32; i++) {
            sum += shared_a[ty * 32 + i] * shared_b[i * 32 + tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}
)";

//ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ OPENCL
#ifdef __APPLE__
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
cl_kernel kernel_simple = NULL;
cl_kernel kernel_block = NULL;
cl_device_id device_id = NULL;
#else
cl_context context = nullptr;
cl_command_queue command_queue = nullptr;
cl_program program = nullptr;
cl_kernel kernel_simple = nullptr;
cl_kernel kernel_block = nullptr;
cl_device_id device_id = nullptr;
#endif
bool opencl_initialized = false;

//Инициализация OpenCL
bool InitOpenCL() {
    if (opencl_initialized) return true;
    cl_int ret;
    cl_uint num_platforms;
    cl_platform_id platform_id = NULL;
    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (ret != CL_SUCCESS || num_platforms == 0) {
        cerr << "Ошибка: Не найдено OpenCL платформ!" << endl;
        return false;
    }
    // Получаем GPU устройство
    cl_uint num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (ret != CL_SUCCESS || num_devices == 0) {
        cerr << "Ошибка: Не найдено GPU устройств для OpenCL!" << endl;
        return false;
    }
    // Получаем информацию об устройстве
    char device_name[256];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    cout << "  OpenCL устройство: " << device_name << endl;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания контекста OpenCL!" << endl;
        return false;
    }
    // Создаём очередь команд
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания очереди команд OpenCL!" << endl;
        return false;
    }
    // Создаём программу из исходного кода
    size_t source_size = strlen(opencl_kernel_source);
    program = clCreateProgramWithSource(context, 1, &opencl_kernel_source, &source_size, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания программы OpenCL!" << endl;
        return false;
    }
    // Компилируем программу
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << "Ошибка компиляции OpenCL кернела:" << endl << log << endl;
        delete[] log;
        return false;
    }
    // Создаём кернелы
    kernel_simple = clCreateKernel(program, "matrixMulKernel", &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания простого кернела!" << endl;
        return false;
    }
    kernel_block = clCreateKernel(program, "matrixMulBlockKernel", &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания блочного кернела!" << endl;
        return false;
    }
    opencl_initialized = true;
    return true;
}
void CleanupOpenCL() {
    if (kernel_simple) clReleaseKernel(kernel_simple);
    if (kernel_block) clReleaseKernel(kernel_block);
    if (program) clReleaseProgram(program);
    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context) clReleaseContext(context);
}
// Умножение OpenCL (GPU)
double* MultiplyWithOpenCL(double* a, double* b, int n, int /*num_threads*/) {
    if (!opencl_initialized) {
        if (!InitOpenCL()) {
            cerr << "Ошибка: Не удалось инициализировать OpenCL!" << endl;
            exit(1);
        }
    }
    double* c = new double[n * n]();
    cl_int ret;
    cl_mem buffer_a, buffer_b, buffer_c;
    buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n * sizeof(double), a, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания буфера A!" << endl;
        delete[] c;
        return nullptr;
    }
    buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  n * n * sizeof(double), b, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания буфера B!" << endl;
        clReleaseMemObject(buffer_a);
        delete[] c;
        return nullptr;
    }
    buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  n * n * sizeof(double), NULL, &ret);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка создания буфера C!" << endl;
        clReleaseMemObject(buffer_a);
        clReleaseMemObject(buffer_b);
        delete[] c;
        return nullptr;
    }
    // Выбираем кернел в зависимости от размера матрицы
    cl_kernel kernel;
    size_t global_work_size[2];
    size_t local_work_size[2];
    if (n >= 512) {
        // Используем оптимизированный кернел с shared memory для больших матриц
        kernel = kernel_block;
        // Настройка размеров для блочного кернела
        local_work_size[0] = 32;
        local_work_size[1] = 32;
        global_work_size[0] = ((n + 31) / 32) * 32;
        global_work_size[1] = ((n + 31) / 32) * 32;
        // Устанавливаем аргументы для блочного кернела
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
        ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
        ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
        ret |= clSetKernelArg(kernel, 3, sizeof(int), &n);
        // Выделяем shared memory для блоков
        size_t shared_mem_size = 32 * 32 * sizeof(double) * 2;
        ret |= clSetKernelArg(kernel, 4, shared_mem_size, NULL);
        ret |= clSetKernelArg(kernel, 5, shared_mem_size, NULL);
    } else {
        // Используем простой кернел для маленьких матриц
        kernel = kernel_simple;
        // Настройка размеров для простого кернела
        local_work_size[0] = 16;
        local_work_size[1] = 16;
        global_work_size[0] = ((n + 15) / 16) * 16;
        global_work_size[1] = ((n + 15) / 16) * 16;
        // Устанавливаем аргументы для простого кернела
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
        ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
        ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
        ret |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    }
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка установки аргументов кернела!" << endl;
        clReleaseMemObject(buffer_a);
        clReleaseMemObject(buffer_b);
        clReleaseMemObject(buffer_c);
        delete[] c;
        return nullptr;
    }
    // Запускаем кернел
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,  global_work_size, local_work_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка запуска кернела!" << endl;
        clReleaseMemObject(buffer_a);
        clReleaseMemObject(buffer_b);
        clReleaseMemObject(buffer_c);
        delete[] c;
        return nullptr;
    }
    // Копируем результат обратно на CPU
    ret = clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0,  n * n * sizeof(double), c, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "Ошибка чтения результата!" << endl;
    }
    // Очищаем буферы
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    return c;
}

//POSIX THREADS
void* InitMatrixThread(void* arguments) {
    InitThreadData* data = (InitThreadData*)arguments;
    mt19937 gen(data->seed);
    uniform_real_distribution<double> dist(data->start_val, data->end_val);
    for (int idx = data->start_idx; idx < data->end_idx; idx++) {
        data->matrix[idx] = dist(gen);
    }
    return NULL;
}
double* InitMatrixWithPthreads(int n, double start, double end, int num_threads) {
    double* matrix = new double[n * n];
    pthread_t threads[num_threads];
    InitThreadData thread_data[num_threads];
    int total_elements = n * n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;
    int current_idx = 0;
    for (int i = 0; i < num_threads; i++) {
        int start_idx = current_idx;
        int end_idx = start_idx + elements_per_thread + (i < remainder ? 1 : 0);
        thread_data[i] = {matrix, start_idx, end_idx, start, end, (unsigned int)(time(NULL) + i)};
        pthread_create(&threads[i], NULL, InitMatrixThread, &thread_data[i]);
        current_idx = end_idx;
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    return matrix;
}

// OpenMP
double* InitMatrixWithOpenMP(int n, double start, double end, int num_threads) {
    double* matrix = new double[n * n];
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num());
        mt19937 gen(seed);
        uniform_real_distribution<double> dist(start, end);
        #pragma omp for
        for (int i = 0; i < n * n; i++) {
            matrix[i] = dist(gen);
        }
    }
    return matrix;
}

//Intel TBB
double* InitMatrixWithTBB(int n, double start, double end, int num_threads) {
    double* matrix = new double[n * n];
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
    parallel_for(blocked_range<int>(0, n * n), [&](const blocked_range<int>& range) {
        static thread_local mt19937 gen(random_device{}());
        uniform_real_distribution<double> dist(start, end);
        for (int i = range.begin(); i < range.end(); i++) {
            matrix[i] = dist(gen);
        }
    });
    return matrix;
}

//Классика
double* InitMatrixSerial(int n, double start, double end) {
    double* matrix = new double[n * n];
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(start, end);
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

//POSIX THREADS
void* MultiplyThreadFunction(void* arguments) {
    MultiplyThreadData* data = (MultiplyThreadData*)arguments;
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->n; j++) {
            double sum = 0.0;
            for (int k = 0; k < data->n; k++) {
                sum += data->a[i * data->n + k] * data->b[k * data->n + j];
            }
            data->c[i * data->n + j] = sum;
        }
    }
    return NULL;
}
double* MultiplyWithPthreads(double* a, double* b, int n, int num_threads) {
    double* c = new double[n * n]();
    pthread_t threads[num_threads];
    MultiplyThreadData thread_data[num_threads];
    int rows_per_thread = n / num_threads;
    int remainder = n % num_threads;
    int current_row = 0;
    for (int i = 0; i < num_threads; i++) {
        int start_row = current_row;
        int end_row = start_row + rows_per_thread + (i < remainder ? 1 : 0);
        thread_data[i] = {a, b, c, n, start_row, end_row};
        pthread_create(&threads[i], NULL, MultiplyThreadFunction, &thread_data[i]);
        current_row = end_row;
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    return c;
}
//OpenMP
double* MultiplyWithOpenMP(double* a, double* b, int n, int num_threads) {
    double* c = new double[n * n]();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
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
//INTEL TBB
double* MultiplyWithTBB(double* a, double* b, int n, int num_threads) {
    double* c = new double[n * n]();
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
    parallel_for(blocked_range<int>(0, n), [&](const blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    });
    return c;
}
double* MultiplySerial(double* a, double* b, int n) {
    double* c = new double[n * n]();
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
double MeasureTime(int size, int num_threads, 
    double* (*init_func)(int, double, double, int),
    double* (*multiply_func)(double*, double*, int, int)) {
    double* A = init_func(size, -10.0, 10.0, num_threads);
    double* B = init_func(size, -10.0, 10.0, num_threads);
    auto start = high_resolution_clock::now();
    double* C = multiply_func(A, B, size, num_threads);
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<milliseconds>(end - start).count();
    delete[] A;
    delete[] B;
    delete[] C;
    return time_ms;
}

double MeasureTimeOpenCL(int size, int num_threads) {
    double* A = InitMatrixSerial(size, -10.0, 10.0);
    double* B = InitMatrixSerial(size, -10.0, 10.0);
    auto start = high_resolution_clock::now();
    double* C = MultiplyWithOpenCL(A, B, size, num_threads);
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<milliseconds>(end - start).count();
    delete[] A;
    delete[] B;
    delete[] C;
    return time_ms;
}
void FindOptimalThreads(int matrix_size) {
    cout << "\nОптимальное число потоков для матрицы " << matrix_size << "x" << matrix_size << endl;
    vector<int> thread_counts = {2, 4, 8, 16};
    
    cout << "Измерение базового времени (1 поток)" << endl;
    double serial_time = MeasureTime(matrix_size, 1, InitMatrixWithOpenMP, MultiplyWithOpenMP);
    cout << "Базовое время: " << fixed << setprecision(2) << serial_time << " мс\n" << endl;
    
    struct MethodResult {
        string name;
        int optimal_threads;
        double max_speedup;
    };
    
    vector<MethodResult> cpu_results;
    vector<MethodResult> all_results;
    
    vector<pair<string, double*(*)(int, double, double, int)>> init_funcs = {
        {"POSIX Threads", InitMatrixWithPthreads},
        {"OpenMP", InitMatrixWithOpenMP},
        {"Intel TBB", InitMatrixWithTBB}
    };
    
    vector<pair<string, double*(*)(double*, double*, int, int)>> mult_funcs = {
        {"POSIX Threads", MultiplyWithPthreads},
        {"OpenMP", MultiplyWithOpenMP},
        {"Intel TBB", MultiplyWithTBB}
    };
    
    for (size_t m = 0; m < init_funcs.size(); m++) {
        MethodResult res;
        res.name = init_funcs[m].first;
        res.optimal_threads = 1;
        res.max_speedup = 1.0;
        cout << "\nТЕСТИРОВАНИЕ МЕТОДА: " << res.name << endl << endl;
        cout << left << setw(12) << "Потоки" << " | " << setw(15) << "Время (мс)" << " | " << setw(15) << "Ускорение" << " | " << setw(15) << "Эффективность" << endl;
        cout << "________________________________________________________________________" << endl;
        for (int threads : thread_counts) {
            if (threads > 32) continue;
            double time = MeasureTime(matrix_size, threads,
                init_funcs[m].second, mult_funcs[m].second);
            double speedup = serial_time / time;
            double efficiency = speedup / threads;
            if (speedup > res.max_speedup) {
                res.max_speedup = speedup;
                res.optimal_threads = threads;
            }
            cout << left << setw(12) << threads << " | " << setw(15) << fixed << setprecision(2) << time
                 << " | " << setw(15) << fixed << setprecision(2) << speedup << "x" << " | " << setw(15) << fixed << setprecision(1) << (efficiency * 100) << "%" << endl;
        }
        cpu_results.push_back(res);
        all_results.push_back(res);
    }
    
    cout << endl;
    
    cout << "\nТЕСТИРОВАНИЕ МЕТОДА: OpenCL (GPU)" << endl << endl;
    cout << left << setw(12) << "Режим" << " | " << setw(15) << "Время (мс)" << " | " << setw(15) << "Ускорение" << " | " << setw(15) << "Примечание" << endl;
    cout << "________________________________________________________________________" << endl;
    bool has_opencl = InitOpenCL();
    if (has_opencl) {
        double opencl_time = MeasureTimeOpenCL(matrix_size, 0);
        double opencl_speedup = serial_time / opencl_time;
        cout << left << setw(12) << "GPU" << " | " << setw(15) << fixed << setprecision(2) << opencl_time
             << " | " << setw(15) << fixed << setprecision(2) << opencl_speedup << "x" << " | " << setw(15) << "GPU ускорение" << endl;
        MethodResult opencl_res;
        opencl_res.name = "OpenCL (GPU)";
        opencl_res.optimal_threads = 0;
        opencl_res.max_speedup = opencl_speedup;
        all_results.push_back(opencl_res);
    } else {
        cout << left << setw(12) << "GPU" << " | " << setw(15) << "N/A" << " | " << setw(15) << "N/A" << " | " << setw(15) << "OpenCL недоступен" << endl;
    }
    
    cout << endl;
    
    cout << "\nОПТИМАЛЬНЫЕ ЗНАЧЕНИЯ ДЛЯ МАТРИЦЫ " << matrix_size << "x" << matrix_size << endl;
    cout << left << setw(25) << "Метод" << " | " << setw(28) << "Оптимальное число потоков" << " | " << "Максимальное ускорение" << endl;
    cout << "________________________________________________________________________________" << endl;
    for (const auto& res : all_results) {
        string threads_str = (res.name == "OpenCL (GPU)") ? "-" : to_string(res.optimal_threads);
        cout << left << setw(25) << res.name << " | " << setw(28) << threads_str << " | " << fixed << setprecision(2) << res.max_speedup << "x" << endl;
    }
    
    // Находим лучший CPU метод (только среди CPU_results, игнорируя OpenCL)
    int best_cpu_idx = 0;
    for (size_t i = 1; i < cpu_results.size(); i++) {
        if (cpu_results[i].max_speedup > cpu_results[best_cpu_idx].max_speedup) {
            best_cpu_idx = i;
        }
    }
    
    // Находим лучший общий метод
    int best_all_idx = 0;
    for (size_t i = 1; i < all_results.size(); i++) {
        if (all_results[i].max_speedup > all_results[best_all_idx].max_speedup) {
            best_all_idx = i;
        }
    }
    
    int num_cores = thread::hardware_concurrency();
    cout << "\nЛучший CPU метод: " << cpu_results[best_cpu_idx].name  << " (ускорение " << fixed << setprecision(2) << cpu_results[best_cpu_idx].max_speedup << "x)" << endl;
    cout << "Оптимальное число потоков для CPU: " << cpu_results[best_cpu_idx].optimal_threads << endl;
    cout << "Рекомендуемое число потоков: " << num_cores << endl;
    
    if (has_opencl && best_all_idx < all_results.size() && all_results[best_all_idx].name == "OpenCL (GPU)") {
        cout << "\nЛучший общий метод: " << all_results[best_all_idx].name  << " (ускорение " << fixed << setprecision(2) << all_results[best_all_idx].max_speedup << "x)" << endl;
    }
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = 0;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
    }
    if (n <= 0) {
        cerr << "Ошибка: неверный размер матрицы!" << endl;
        return 1;
    }
    if (num_threads <= 0 || num_threads > 32) {
        cerr << "Ошибка: неверное количество потоков!" << endl;
        return 1;
    }
    
    cout << "ПРОГРАММА УМНОЖЕНИЯ МАТРИЦ С МНОГОПОТОЧНОЙ РЕАЛИЗАЦИЕЙ" << endl;
    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Количество потоков (CPU): " << num_threads << endl;
    
    bool has_opencl = false;
    cout << "Проверка OpenCL" << endl;
    has_opencl = InitOpenCL();
    if (has_opencl) {
        cout << "OpenCL будет использовать GPU для умножения" << endl;
    } else {
        cout << "OpenCL не доступен или GPU не обнаружены" << endl;
    }
    cout << endl;
    

    cout << "Классический однопоточный метод" << endl;
    double* A_serial = InitMatrixSerial(n, -10.0, 10.0);
    double* B_serial = InitMatrixSerial(n, -10.0, 10.0);
    auto start_serial = high_resolution_clock::now();
    double* C_serial = MultiplySerial(A_serial, B_serial, n);
    auto end_serial = high_resolution_clock::now();
    double serial_time = duration_cast<milliseconds>(end_serial - start_serial).count();
    cout << "Время выполнения: " << fixed << setprecision(2) << serial_time << " мс" << endl;
    delete[] A_serial;
    delete[] B_serial;
    delete[] C_serial;
    

    cout << "\nPOSIX Threads" << endl;
    double pthread_time = MeasureTime(n, num_threads, InitMatrixWithPthreads, MultiplyWithPthreads);
    double pthread_speedup = serial_time / pthread_time;
    cout << "  Время выполнения: " << fixed << setprecision(2) << pthread_time << " мс" << endl;
    cout << "  Ускорение: " << fixed << setprecision(2) << pthread_speedup << "x" << endl;
    

    cout << "\nOpenMP" << endl;
    double omp_time = MeasureTime(n, num_threads, InitMatrixWithOpenMP, MultiplyWithOpenMP);
    double omp_speedup = serial_time / omp_time;
    cout << "  Время выполнения: " << fixed << setprecision(2) << omp_time << " мс" << endl;
    cout << "  Ускорение: " << fixed << setprecision(2) << omp_speedup << "x" << endl;
    

    cout << "\nIntel TBB " << endl;
    double tbb_time = MeasureTime(n, num_threads, InitMatrixWithTBB, MultiplyWithTBB);
    double tbb_speedup = serial_time / tbb_time;
    cout << "  Время выполнения: " << fixed << setprecision(2) << tbb_time << " мс" << endl;
    cout << "  Ускорение: " << fixed << setprecision(2) << tbb_speedup << "x" << endl;
    

    cout << "\nOpenCL (GPU умножение)" << endl;
    if (has_opencl) {
        try {
            double* A_gpu = InitMatrixSerial(n, -10.0, 10.0);
            double* B_gpu = InitMatrixSerial(n, -10.0, 10.0);
            auto start_gpu = high_resolution_clock::now();
            double* C_gpu = MultiplyWithOpenCL(A_gpu, B_gpu, n, num_threads);
            auto end_gpu = high_resolution_clock::now();
            double opencl_time = duration_cast<milliseconds>(end_gpu - start_gpu).count();
            double opencl_speedup = serial_time / opencl_time;
            cout << "  Время выполнения (GPU): " << fixed << setprecision(2) << opencl_time << " мс" << endl;
            cout << "  Ускорение (GPU vs CPU): " << fixed << setprecision(2) << opencl_speedup << "x" << endl;
            delete[] A_gpu;
            delete[] B_gpu;
            delete[] C_gpu;
        } catch (const exception& e) {
            cout << "  Ошибка при выполнении OpenCL: " << e.what() << endl;
        }
    } else {
        cout << "  OpenCL не доступен. Умножение на GPU невозможно." << endl;
    }

    cout << "\nСравнение результатов умножения:" << endl;
    cout << "  POSIX Threads: " << fixed << setprecision(2) << pthread_time  << " мс (ускорение " << pthread_speedup << "x)" << endl;
    cout << "  OpenMP:        " << fixed << setprecision(2) << omp_time << " мс (ускорение " << omp_speedup << "x)" << endl;
    cout << "  Intel TBB:     " << fixed << setprecision(2) << tbb_time << " мс (ускорение " << tbb_speedup << "x)" << endl;
    
    if (has_opencl) {
        double* A_gpu = InitMatrixSerial(n, -10.0, 10.0);
        double* B_gpu = InitMatrixSerial(n, -10.0, 10.0);
        auto start_gpu = high_resolution_clock::now();
        double* C_gpu = MultiplyWithOpenCL(A_gpu, B_gpu, n, num_threads);
        auto end_gpu = high_resolution_clock::now();
        double opencl_time = duration_cast<milliseconds>(end_gpu - start_gpu).count();
        double opencl_speedup = serial_time / opencl_time;
        cout << "  OpenCL (GPU):  " << fixed << setprecision(2) << opencl_time 
             << " мс (ускорение " << opencl_speedup << "x)" << endl;
        delete[] A_gpu;
        delete[] B_gpu;
        delete[] C_gpu;
    }
    cout << "  Последовательно: " << fixed << setprecision(2) << serial_time << " мс" << endl;

    double best_speedup = pthread_speedup;
    string best_method = "POSIX Threads";
    if (omp_speedup > best_speedup) {
        best_speedup = omp_speedup;
        best_method = "OpenMP";
    }
    if (tbb_speedup > best_speedup) {
        best_speedup = tbb_speedup;
        best_method = "Intel TBB";
    }
    cout << "\nЛучший CPU метод: " << best_method << " (ускорение " << best_speedup << "x)" << endl;
    if (has_opencl) {
        double* A_gpu = InitMatrixSerial(n, -10.0, 10.0);
        double* B_gpu = InitMatrixSerial(n, -10.0, 10.0);
        auto start_gpu = high_resolution_clock::now();
        double* C_gpu = MultiplyWithOpenCL(A_gpu, B_gpu, n, num_threads);
        auto end_gpu = high_resolution_clock::now();
        double opencl_time = duration_cast<milliseconds>(end_gpu - start_gpu).count();
        double opencl_speedup = serial_time / opencl_time;
        if (opencl_speedup > best_speedup) {
            cout << "OpenCL (GPU) показывает лучшее ускорение: " << fixed << setprecision(2) << opencl_speedup << "x" << endl;
        }
        delete[] A_gpu;
        delete[] B_gpu;
        delete[] C_gpu;
    }

    vector<int> optimal_sizes = {500, 1000, 2000};
    cout << "\nПОИСК ОПТИМАЛЬНОГО ЧИСЛА ПОТОКОВ ДЛЯ РАЗНЫХ РАЗМЕРОВ" << endl;
    for (int size : optimal_sizes) {
        if (size <= n * 2) {
            FindOptimalThreads(size);
        }
    }
    // Очистка OpenCL
    if (has_opencl) {
        CleanupOpenCL();
    }
    return 0;
}

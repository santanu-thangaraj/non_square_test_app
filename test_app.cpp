
#include <iostream>
#include <CL/cl.h>
#include "test_app.h"

#define AVAIL_MEM_SIZE 32768

template<typename T, typename C>
class test_inplace_transpose
{
public:
    test_inplace_transpose(char** argv,
                           cl_context context,
                           int L,
                           int M);

   void inplace_transpose_square_CPU(int start_x, T* z);

   void inplace_transpose_swap_CPU(T* z);
   int inplace_transpose_swap_GPU(char** argv,
                                   cl_command_queue  commands,
                                   cl_kernel kernel,
                                   cl_double *ptr_NDRangePureExecTimeMs);

   int inplace_transpose_square_GPU( char** argv,
                                      cl_command_queue  commands,
                                      cl_kernel kernel,
                                      cl_double *ptr_NDRangePureExecTimeMs
                                     );
   int verify_initial_trans(T* input, T* output);

   void verify_initial_trans_CPU();
   void verify_initial_trans_GPU();

   int verify_swap(T* input, T* output);

   void snake(T *ts, T *td, int num_lines_loaded, int is, int id, int pos, T* z);
   void verify_swap_CPU();
   void verify_swap_GPU();

   ~test_inplace_transpose();
   int is_complex_planar;
   int start_inx;
   cl_mem input_output_buffer_device;
   T* x; //original data
   T* y; // GPU output
   T* z; // CPU output
   T* interm; //CPU intermediate data

   cl_mem input_output_buffer_device_R;
   cl_mem input_output_buffer_device_I;
   T* x_R;
   T* x_I;
   T* y_R;
   T* y_I;
   T* z_R;
   T* z_I;
   T* interm_R;
   T* interm_I;

private:

    int L;
    int M;
    int small_dim;
    int big_dim;
    int width;  
    size_t local_work_size_swap;
    size_t global_work_size_trans;
};

template<typename T, typename C>
void test_inplace_transpose<T,C>::snake(T *ts, T *td, int num_lines_loaded, int is, int id, int pos, T* z)
{
    // pos 0 - start, 1 - inter, 2 - end

    for (int p = 0; p < num_lines_loaded; p++)
    {
        for (int j = 0; j < small_dim; j++)
        {
            if (pos == 0)
            {
                ts[p*small_dim + j] = z[is*num_lines_loaded*small_dim + p*small_dim + j];
                td[p*small_dim + j] = z[id*num_lines_loaded*small_dim + p*small_dim + j];
                z[id*num_lines_loaded*small_dim + p*small_dim + j] = ts[p*small_dim + j];
            }
            else if (pos == 1)
            {
                td[p*small_dim + j] = z[id*num_lines_loaded*small_dim + p*small_dim + j];
                z[id*num_lines_loaded*small_dim + p*small_dim + j] = ts[p*small_dim + j];
            }
            else
            {
                z[id*num_lines_loaded*small_dim + p*small_dim + j] = ts[p*small_dim + j];
            }
        }
    }
}
/* -> get_cycles funcition gets the swapping logic required for given row x col matrix.
-> cycle_map[0] holds the total number of cycles required.
-> cycles start and end with the same index, hence we can identify individual cycles,
though we tend to store the cycle index contiguously*/
void get_cycles(int *cycle_map, int num_reduced_row, int num_reduced_col)
{
    int *is_swapped = new int[num_reduced_row * num_reduced_col];
    int i, map_index = 1, num_cycles = 0;
    int swap_id;
    /*initialize swap map*/
    is_swapped[0] = 1;
    is_swapped[num_reduced_row * num_reduced_col - 1] = 1;
    for (i = 1; i < (num_reduced_row * num_reduced_col - 1); i++)
    {
        is_swapped[i] = 0;
    }

    for (i = 1; i < (num_reduced_row * num_reduced_col - 1); i++)
    {
        swap_id = i;
        while (!is_swapped[swap_id])
        {
            is_swapped[swap_id] = 1;
            cycle_map[map_index++] = swap_id;
            swap_id = (num_reduced_row * swap_id) % (num_reduced_row * num_reduced_col - 1);
            if (swap_id == i)
            {
                cycle_map[map_index++] = swap_id;
                num_cycles++;
            }
        }
    }
    cycle_map[0] = num_cycles;
}

/* This function factorizes L and it finds a maximum of
the 'factors less than max_capacity'*/
int get_num_lines_to_be_loaded(int max_capacity, int L)
{
    if (L < max_capacity)
    {
        return L;
    }

    int square_root = (int)sqrt(L) + 1;
    int max_factor = 1;
    for (int i = 1; i < square_root; i++)
    {
        if (L % i == 0)
        {
            if ((i > max_factor) && (i <= max_capacity))
            {
                max_factor = i;
            }

            if (((L / i) > max_factor) && ((L / i) <= max_capacity))
            {
                max_factor = L / i;
            }
        }
    }
    return max_factor;
}

template<typename T, typename C>
int test_inplace_transpose<T, C>::inplace_transpose_swap_GPU(char** argv,
                                                                cl_command_queue  commands,
                                                                cl_kernel kernel_swap,
                                                                cl_double *ptr_NDRangePureExecTimeMs)
{
    cl_int status;
    size_t global_work_size[MAX_DIMS] = { 1,1,1 };
    size_t local_work_size[MAX_DIMS] = { 1,1,1 };
    size_t num_work_items_in_group = local_work_size_swap;
    int tot_num_work_items = num_work_items_in_group;
    cl_event warmup_event;
    cl_event perf_event[NUM_PERF_ITERATIONS];
    /*size_t global_work_offset[MAX_DIMS] = {0,0,0};*/
    cl_ulong start = 0, end = 0;
    int i;
    cl_double perf_nums[NUM_PERF_ITERATIONS];

    global_work_size[0] = tot_num_work_items;
    local_work_size[0] = num_work_items_in_group;

    if (!is_complex_planar)
    {
        status = clSetKernelArg(kernel_swap,
            0,
            sizeof(cl_mem),
            &input_output_buffer_device);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }
    }
    else
    {
        status = clSetKernelArg(kernel_swap,
            0,
            sizeof(cl_mem),
            &input_output_buffer_device_R);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }

        status = clSetKernelArg(kernel_swap,
            1,
            sizeof(cl_mem),
            &input_output_buffer_device_I);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }
    }

    /*This is a warmup run*/
    status = clEnqueueNDRangeKernel(commands,
        kernel_swap,
        SUPPORTED_WORK_DIM,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        &warmup_event);

    if (status != CL_SUCCESS)
    {
        std::cout << "The kernel launch failure\n";
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &warmup_event);

    status = clReleaseEvent(warmup_event);
    if (status != CL_SUCCESS)
    {
        std::cout << "Error releasing events\n";
        return EXIT_FAILURE;
    }

    if (!is_complex_planar)
    {
        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device,
            CL_TRUE,
            0,
            L * M * sizeof(T),
            y,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }
    }
    else
    {
        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device_R,
            CL_TRUE,
            0,
            L * M * sizeof(T),
            y_R,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }

        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device_I,
            CL_TRUE,
            0,
            L * M * sizeof(T),
            y_I,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }
    }
    *ptr_NDRangePureExecTimeMs = 0;
#if 0
    for (i = 0; i < NUM_PERF_ITERATIONS; i++)
    {
        status = clEnqueueNDRangeKernel(commands,
            kernel,
            SUPPORTED_WORK_DIM,
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            &perf_event[i]);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel launch failure\n";
            return EXIT_FAILURE;
        }
        clWaitForEvents(1, &perf_event[i]);

        clGetEventProfilingInfo(perf_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(perf_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

        status = clReleaseEvent(perf_event[i]);
        if (status != CL_SUCCESS)
        {
            std::cout << "Error releasing events\n";
            return EXIT_FAILURE;
        }

        /*the resolution of the events is 1e-09 sec*/
        perf_nums[i] = (cl_double)(end - start)*(cl_double)(1e-06);
    }

    /*Take the median of the performance numbers*/
    std::sort(perf_nums, (perf_nums + NUM_PERF_ITERATIONS));
    *ptr_NDRangePureExecTimeMs = perf_nums[(NUM_PERF_ITERATIONS + 1) / 2];
#endif
    return EXIT_SUCCESS;
}
template<typename T, typename C>
void test_inplace_transpose<T, C>::inplace_transpose_swap_CPU(T* z)
{
    size_t avail_mem = AVAIL_MEM_SIZE / sizeof(T);
    if (is_complex_planar)
    {
        avail_mem /= 2;
    }
    T *full_mem = new T[avail_mem];
    T *te = full_mem; //= new T[avail_mem >> 1];
    T *to = full_mem + (avail_mem >> 1); //= new T[avail_mem >> 1];
    int max_capacity = (avail_mem >> 1) / small_dim;
    if (max_capacity <= 0)
    {
        std::cout << "\nIn-place transpose cannot be performed within specified memory constraints.\n";
        exit(1);
    }
    int num_lines_loaded = get_num_lines_to_be_loaded(max_capacity, small_dim);
    int num_reduced_row; 
    int num_reduced_col;

    local_work_size_swap = num_lines_loaded << 4;
    local_work_size_swap = (local_work_size_swap > 256) ? 256 : local_work_size_swap;

    size_t num_threads_processing_row = (256 / local_work_size_swap) * 16;
    local_work_size_swap = num_lines_loaded * num_threads_processing_row;

    if (L == small_dim)
    {
        num_reduced_row = std::ceil((float)small_dim / (float)(num_lines_loaded));
        num_reduced_col = 2;
    }
    else
    {
        num_reduced_row = 2;
        num_reduced_col = std::ceil((float)small_dim / (float)(num_lines_loaded)); 
    }

    /* The reduced row and col comes from the fact that we perform swaps of 'num_lines_loaded' rows first
    and thus reducing the amount of swaps required to be done using the snake function*/
    int i;

    if (num_lines_loaded > 1)
    {
        if (L == small_dim)
        {
            for (i = 0; i < big_dim; i += 2 * num_lines_loaded)
            {
                // read
                for (int p = 0; p < num_lines_loaded; p++)
                {
                    for (int j = 0; j < small_dim; j++)
                    {
                        te[p*small_dim + j] = z[i*small_dim + (2 * p + 0)*small_dim + j];
                        to[p*small_dim + j] = z[i*small_dim + (2 * p + 1)*small_dim + j];
                    }
                }

                // write
                for (int p = 0; p < num_lines_loaded; p++)
                {
                    for (int j = 0; j < small_dim; j++)
                    {
                        z[i*small_dim + 0 + p*small_dim + j] = te[p*small_dim + j];
                        z[i*small_dim + num_lines_loaded*small_dim + p*small_dim + j] = to[p*small_dim + j];
                    }
                }
            }
        }
        else
        {

            for (i = 0; i < small_dim; i += num_lines_loaded)
            {
                // read
                for (int p = 0; p < num_lines_loaded; p++)
                {
                    for (int j = 0; j < small_dim; j++)
                    {
                        full_mem[(2 * p) * small_dim + j] = z[i*small_dim + p*small_dim + j];
                        full_mem[(2 * p + 1) * small_dim + j] = z[small_dim * small_dim + i*small_dim + p*small_dim + j];
                    }
                }

                // write
                for (int p = 0; p < num_lines_loaded; p++)
                {
                    for (int j = 0; j < small_dim; j++)
                    {
                        z[i*small_dim + p*small_dim + j] = full_mem[p * small_dim + j];
                        z[small_dim * small_dim + i*small_dim + p*small_dim + j] = full_mem[(num_lines_loaded + p) * small_dim + j];

                    }
                }
            }
        }
    }
    //memcpy(interm,z,L*M*sizeof(T));
    int *cycle_map = new int[num_reduced_row * num_reduced_col * 2];
    /* The memory required by cycle_map canniot exceed 2 times row*col by design*/

    get_cycles(cycle_map, num_reduced_row, num_reduced_col);

    T *ta = te;
    T *tb = to;
    T *ttmp;

    int inx = 0, start_inx;
    for (i = 0; i < cycle_map[0]; i++)
    {
        start_inx = cycle_map[++inx];
        std::cout << "\nCycle:" << (i + 1) << ">\t" << "(" << start_inx << "," << cycle_map[inx + 1] << ")";
        snake(ta, tb, num_lines_loaded, start_inx, cycle_map[inx + 1], 0, z);

        while (start_inx != cycle_map[++inx])
        {
            ttmp = ta;
            ta = tb;
            tb = ttmp;

            std::cout << "\t" << "(" << cycle_map[inx] << "," << cycle_map[inx + 1] << ")";
            int action_var = (cycle_map[inx + 1] == start_inx) ? 2 : 1;
            snake(ta, tb, num_lines_loaded, cycle_map[inx], cycle_map[inx + 1], action_var, z);
        }
    }

    delete[] full_mem;
//    delete[] to;
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_initial_trans_CPU()
{
    if (!is_complex_planar)
    {
        this->verify_initial_trans(x, z);
    }
    else
    {
        this->verify_initial_trans(x_R, z_R);
        this->verify_initial_trans(x_I, z_I);
    }
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_initial_trans_GPU()
{
    if (!is_complex_planar)
    {
        this->verify_initial_trans(x, y);
    }
    else
    {
        this->verify_initial_trans(x_R, y_R);
        this->verify_initial_trans(x_I, y_I);
    }
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_swap_CPU()
{
    if (!is_complex_planar)
    {
        this->verify_swap(x, z);
    }
    else
    {
        this->verify_swap(x_R, z_R);
        this->verify_swap(x_I, z_I);
    }
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_swap_GPU()
{
    if (!is_complex_planar)
    {
        this->verify_swap(x, y);
    }
    else
    {
        this->verify_swap(x_R, y_R);
        this->verify_swap(x_I, y_I);
    }
}

template<typename T, typename C>
int test_inplace_transpose<T, C>::inplace_transpose_square_GPU(char** argv,
                                                                cl_command_queue  commands,
                                                                cl_kernel kernel,
                                                                cl_double *ptr_NDRangePureExecTimeMs
                                                              )
{
    cl_int status;
    size_t global_work_size[MAX_DIMS] = { 1,1,1 };
    size_t local_work_size[MAX_DIMS] = { 1,1,1 };
    size_t num_work_items_in_group = 256;
    int tot_num_work_items = global_work_size_trans;
    cl_event warmup_event;
    cl_event perf_event[NUM_PERF_ITERATIONS];
    /*size_t global_work_offset[MAX_DIMS] = {0,0,0};*/
    cl_ulong start = 0, end = 0;
    int i;
    cl_double perf_nums[NUM_PERF_ITERATIONS];

    global_work_size[0] = tot_num_work_items;
    local_work_size[0] = num_work_items_in_group;
    if (!is_complex_planar)
    {
        status = clSetKernelArg(kernel,
            0,
            sizeof(cl_mem),
            &input_output_buffer_device);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }
    }
    else
    {
        status = clSetKernelArg(kernel,
            0,
            sizeof(cl_mem),
            &input_output_buffer_device_R);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }

        status = clSetKernelArg(kernel,
            1,
            sizeof(cl_mem),
            &input_output_buffer_device_I);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel set argument failure\n";
            return EXIT_FAILURE;
        }
    }
    /*This is a warmup run*/
    status = clEnqueueNDRangeKernel(commands,
        kernel,
        SUPPORTED_WORK_DIM,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        &warmup_event);

    if (status != CL_SUCCESS)
    {
        std::cout << "The kernel launch failure\n";
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &warmup_event);

    status = clReleaseEvent(warmup_event);
    if (status != CL_SUCCESS)
    {
        std::cout << "Error releasing events\n";
        return EXIT_FAILURE;
    }

    if (!is_complex_planar)
    {
        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device,
            CL_TRUE,
            0,
            L * M * sizeof(T),
            y,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }
    }
    else
    {
        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device_R,
            CL_TRUE,
            0,
            L * M * sizeof(C),
            y_R,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }

        status = clEnqueueReadBuffer(commands,
            input_output_buffer_device_I,
            CL_TRUE,
            0,
            L * M * sizeof(C),
            y_I,
            0,
            NULL,
            NULL);

        if (status != CL_SUCCESS)
        {
            std::cout << "Error Reading output buffer.\n";
            return EXIT_FAILURE;
        }
    }
    *ptr_NDRangePureExecTimeMs = 0;
#if 0
    for (i = 0; i < NUM_PERF_ITERATIONS; i++)
    {
        status = clEnqueueNDRangeKernel(commands,
            kernel,
            SUPPORTED_WORK_DIM,
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            &perf_event[i]);

        if (status != CL_SUCCESS)
        {
            std::cout << "The kernel launch failure\n";
            return EXIT_FAILURE;
        }
        clWaitForEvents(1, &perf_event[i]);

        clGetEventProfilingInfo(perf_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(perf_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

        status = clReleaseEvent(perf_event[i]);
        if (status != CL_SUCCESS)
        {
            std::cout << "Error releasing events\n";
            return EXIT_FAILURE;
        }

        /*the resolution of the events is 1e-09 sec*/
        perf_nums[i] = (cl_double)(end - start)*(cl_double)(1e-06);
    }

    /*Take the median of the performance numbers*/
    std::sort(perf_nums, (perf_nums + NUM_PERF_ITERATIONS));
    *ptr_NDRangePureExecTimeMs = perf_nums[(NUM_PERF_ITERATIONS + 1) / 2];
#endif
    return EXIT_SUCCESS;
};

template<typename T, typename C>
int test_inplace_transpose<T, C>::verify_swap(T* input, T* output)
{
    for (int l = 0; l < L; l++)
        for (int m = 0; m < M; m++)
        {
            C* ptr_tmp_input = (C*)(&input[l * M + m]);
            C* ptr_tmp_output = (C*)(&output[m * L + l]);

            for (int comp = 0; comp < (sizeof(T) / sizeof(C)); comp++)
            {
                if (ptr_tmp_output[comp] != ptr_tmp_input[comp])
                {
                    std::cout << "fail" << std::endl;
                    return -1;
                }
            }

        }
}

template<typename T, typename C>
int test_inplace_transpose<T, C>::verify_initial_trans(T* input, T* output)
{

    for (int l = 0; l < small_dim; l++)
        for (int w = 0; w < small_dim; w++)
        {
            for (int comp = 0; comp < (sizeof(T) / sizeof(C)); comp++)
            {
                C* ptr_tmp_input = (C*)(&input[l + w * width]);
                C* ptr_tmp_output = (C*)(&output[l * width + w]);

                if (ptr_tmp_input[comp] != ptr_tmp_output[comp])
                {
                    std::cout << "Fail\n";
                    return 1;
                }

                ptr_tmp_input = (C*)(&input[l + w * width + start_inx]);
                ptr_tmp_output = (C*)(&output[l * width + w + start_inx]);

                if (ptr_tmp_input[comp] != ptr_tmp_output[comp])
                {
                    std::cout << "Fail\n";
                    return 1;
                }
            }
        }
    

    return 0;
};

template<typename T, typename C>
test_inplace_transpose<T,C>::test_inplace_transpose(char** argv,
                                               cl_context context,
                                               int inp_L,
                                               int inp_M)
{
    cl_int status;
    
    L = inp_L;
    M = inp_M;

    small_dim = (L < M) ? L : M;
    big_dim = small_dim * 2;
    width = M;

    start_inx = (small_dim == L) ? L : M*M;

    if (strcmp(argv[1], "P2P") == 0)
    {
        is_complex_planar = 1;
    }
    else
    {
        is_complex_planar = 0;
    }
    int reShapeFactor = 2, wg_slice;
    if (small_dim % (16 * reShapeFactor) == 0)
        wg_slice = small_dim / 16 / reShapeFactor;
    else
        wg_slice = (small_dim / (16 * reShapeFactor)) + 1;

    global_work_size_trans = wg_slice*(wg_slice + 1) / 2 * 16 * 16;
    global_work_size_trans *= 2;

    if (!is_complex_planar)
    {
        x = new T[L*M];
        y = new T[L*M];
        z = new T[L*M];
        interm = new T[L*M];
    }
    else
    {
        x_R = new T[L*M];
        x_I = new T[L*M];
        y_R = new T[L*M];
        y_I = new T[L*M];
        z_R = new T[L*M];
        z_I = new T[L*M];
        interm_R = new T[L*M];
        interm_I = new T[L*M];
    }

    if (!is_complex_planar)
    {
        for (int l = 0; l < L; l++)
            for (int m = 0; m < M; m++)
            {
                C* ptr_tmp_x = (C*)(&x[l*M + m]);
                ptr_tmp_x[0] = rand() % 1000;
                if ((sizeof(T) / sizeof(C)) == 2)
                {
                    ptr_tmp_x[1] = rand() % 1000;
                }
            }
    }
    else
    {
        for (int l = 0; l < L; l++)
            for (int m = 0; m < M; m++)
            {
                C* ptr_tmp_x = (C*)(&x_R[l*M + m]);
                ptr_tmp_x[0] = rand() % 1000;
                ptr_tmp_x = (C*)(&x_I[l*M + m]);
                ptr_tmp_x[0] = rand() % 1000;
            }
    }

    if (!is_complex_planar)
    {

        input_output_buffer_device = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            L * M * sizeof(T),
            x,
            &status);

        if (status != CL_SUCCESS)
        {
            std::cout << "ERROR! Allocation of GPU input buffer failed.\n";
        }
    }
    else
    {
        input_output_buffer_device_R = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            L * M * sizeof(T),
            x_R,
            &status);

        if (status != CL_SUCCESS)
        {
            std::cout << "ERROR! Allocation of GPU input buffer failed.\n";
        }

        input_output_buffer_device_I = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            L * M * sizeof(T),
            x_I,
            &status);

        if (status != CL_SUCCESS)
        {
            std::cout << "ERROR! Allocation of GPU input buffer failed.\n";
        }
    }

   /* for (int l = 0; l<L; l++)
        for (int m = 0; m<M; m++)
        {
            C* ptr_tmp_x = (C*)(&x[l*M + m]);
            C* ptr_tmp_y = (C*)(&y[m*L + l]);
            ptr_tmp_y[0] = ptr_tmp_x[0];
            if ((sizeof(T) / sizeof(C)) == 2)
            {
                ptr_tmp_y[1] = ptr_tmp_x[1];
            }
        }*/
    if (!is_complex_planar)
    {
        memcpy(z, x, L*M*sizeof(T));
    }
    else
    {
        memcpy(z_R, x_R, L*M*sizeof(T));
        memcpy(z_I, x_I, L*M*sizeof(T));
    }
}

template<typename T, typename C>
test_inplace_transpose<T, C>::~test_inplace_transpose()
{
    if (!is_complex_planar)
    {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] interm;
    }
    else
    {
        delete[] x_R;
        delete[] y_R;
        delete[] z_R;
        delete[] interm_R;
        delete[] x_I;
        delete[] y_I;
        delete[] z_I;
        delete[] interm_I;
    }
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::inplace_transpose_square_CPU(int start_x, T* z)
{

    for (int l = 0; l < small_dim; l++)
        for (int m = 0; m < small_dim; m++)
        {
            if (m < l)
                continue;

            T tmp = z[l*width + m + start_x];
            z[l*width + m + start_x] = z[m*width + l + start_x];
            z[m*width + l + start_x] = tmp;
        }
    
}

int opencl_build_kernel(char** argv,
    cl_context* ptr_context,
    cl_command_queue* ptr_commands,
    cl_kernel* ptr_kernel_ST,
    cl_kernel* ptr_kernel_swap,
    INPUT_OUTPUT_TYPE_T input_output_type)
{
    cl_uint             numPlatforms;
    cl_platform_id      platform;
    cl_device_id        device_id;
    cl_program          program;

    int err;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Getting Platforms. (clGetPlatformsIDs)\n";

    }

    if (numPlatforms > 0)
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        err = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (err != CL_SUCCESS)
        {
            std::cout << "Error: Getting Platform Ids. (clGetPlatformsIDs)\n";

        }
        for (unsigned int i = 0; i < numPlatforms; ++i)
        {
            char pbuff[100];
            err = clGetPlatformInfo(
                platforms[i],
                CL_PLATFORM_VENDOR,
                sizeof(pbuff),
                pbuff,
                NULL);
            if (err != CL_SUCCESS)
            {
                std::cout << "Error: Getting Platform Info.(clGetPlatformInfo)\n";
            }
            platform = platforms[i];
            if (!std::strcmp(pbuff, "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
        delete platforms;
    }

    if (NULL == platform)
    {
        std::cout << "NULL platform found so Exiting Application." << std::endl;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create a device group!\n";
        return EXIT_FAILURE;
    }

    /*Create a compute context*/
    *ptr_context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!(*ptr_context))
    {
        std::cout << "Error: Failed to create a compute context!\n";
        return EXIT_FAILURE;
    }
    /* Create a command commands*/

    *ptr_commands = clCreateCommandQueue(*ptr_context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!(*ptr_commands))
    {
        std::cout << "Error: Failed to create a command commands!\n";
        return EXIT_FAILURE;
    }

    /*Building Kernel*/
    const char * filename = argv[2];
    char         full_path[100];
#ifdef OS_LINUX
    strcpy(full_path, "./kernel_files/");
#else
    strcpy(full_path, ".\\kernel_files\\");
#endif
    strcat(full_path, filename);

    char *src;
    FILE *file = fopen(full_path, "rb");
    size_t srcsize, f_err;
    if (file == NULL)
    {
        std::cout << "\nFile containing the kernel code(\".cl\") not found.\n";
    }

    /* obtain file size:*/
    int fd = fileno(file); //if you have a stream (e.g. from fopen), not a file descriptor.
    struct stat buf;
    fstat(fd, &buf);
    srcsize = buf.st_size;

    // allocate memory to contain the whole file:
    src = (char*)malloc(sizeof(char)*srcsize);
    if (src == NULL)
    {
        std::cout << "\nMemory allocation failed.\n";
        return EXIT_FAILURE;
    }

    f_err = fread(src, 1, srcsize, file);
    if (f_err != srcsize)
    {
        std::cout << "\nfread failed.\n";
        perror("The following error occurred");
        return EXIT_FAILURE;
    }
    fclose(file);

    const char *srcptr[] = { src };

    program = clCreateProgramWithSource(
        *ptr_context,
        1,
        srcptr,
        &srcsize,
        &err);

    if (err != CL_SUCCESS)
    {
        std::cout <<
            "Error: Loading Binary into cl_program \
                           (clCreateProgramWithBinary)\n";
        return EXIT_FAILURE;
    }


    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        std::cout << "Error: Failed to build program executable!\n";
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << buffer << "\n";
        return EXIT_FAILURE;
    }

    /*Create the compute kernel in the program we wish to run*/

    *ptr_kernel_ST = clCreateKernel(program, "transpose_nonsquare", &err);
    if (!(*ptr_kernel_ST) || err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create kernel!\n";
        return EXIT_FAILURE;
    }
    
    *ptr_kernel_swap = clCreateKernel(program, "swap_nonsquare", &err);
    if (!(*ptr_kernel_swap) || err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create kernel!\n";
        return EXIT_FAILURE;
    }

    err = clReleaseProgram(program);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error releasing program\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;

};

template<typename T, typename C>
int master_test_function(char** argv, cl_context context, cl_command_queue commands, cl_kernel kernel_ST, cl_kernel kernel_swap)
{
    const int L = atoi(argv[5]);
    const int M = atoi(argv[4]);
    cl_double           NDRangePureExecTimeMs;

    test_inplace_transpose<T, C> test_inplace_transpose(argv, context, L, M);

    if (!test_inplace_transpose.is_complex_planar)
    {
        test_inplace_transpose.inplace_transpose_square_CPU(0, test_inplace_transpose.z);
        test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx, test_inplace_transpose.z);
    }
    else
    {
        test_inplace_transpose.inplace_transpose_square_CPU(0, test_inplace_transpose.z_R);
        test_inplace_transpose.inplace_transpose_square_CPU(0, test_inplace_transpose.z_I);
        test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx, test_inplace_transpose.z_R);
        test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx, test_inplace_transpose.z_I);
    }

    test_inplace_transpose.inplace_transpose_square_GPU(argv, commands, kernel_ST, &NDRangePureExecTimeMs);

    test_inplace_transpose.verify_initial_trans_CPU();
    test_inplace_transpose.verify_initial_trans_GPU();

    if (!test_inplace_transpose.is_complex_planar)
    {
        test_inplace_transpose.inplace_transpose_swap_CPU(test_inplace_transpose.z);
    }
    else
    {
        test_inplace_transpose.inplace_transpose_swap_CPU(test_inplace_transpose.z_R);
        test_inplace_transpose.inplace_transpose_swap_CPU(test_inplace_transpose.z_I);
    }

    test_inplace_transpose.verify_swap_CPU();

    test_inplace_transpose.inplace_transpose_swap_GPU(argv, commands, kernel_swap, &NDRangePureExecTimeMs);
    //test_inplace_transpose.verify_interm_swap_GPU();
    test_inplace_transpose.verify_swap_GPU();

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    int status;

    cl_context          context;
    cl_kernel           kernel_ST;
    cl_kernel           kernel_swap;
    cl_command_queue    commands;

    enum INPUT_OUTPUT_TYPE_T input_output_type;

    input_output_type = COMPLEX_TO_COMPLEX;

    if (strcmp(argv[1], "R2R") == 0)
    {
        input_output_type = REAL_TO_REAL;
    }

    if (strcmp(argv[1], "P2P") == 0)
    {
        input_output_type = PLANAR_TO_PLANAR;
    }
    /*initialize the command queues and build kernel*/
    status = opencl_build_kernel(argv,
        &context,
        &commands,
        &kernel_ST,
        &kernel_swap,
        input_output_type);

    if (status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    if (input_output_type == COMPLEX_TO_COMPLEX)
    {
        if (strcmp(argv[3], "float") == 0)
        {
            master_test_function<cl_float2,cl_float>(argv, context, commands, kernel_ST, kernel_swap);
        }
        else
        {
            master_test_function<cl_double2, cl_double>(argv, context, commands, kernel_ST, kernel_swap);
        }
    }
    else
    {
        if (strcmp(argv[3], "float") == 0)
        {
            master_test_function<cl_float, cl_float>(argv, context, commands, kernel_ST, kernel_swap);

        }
        else
        {
            master_test_function<cl_double, cl_double>(argv, context, commands, kernel_ST, kernel_swap);

        }
    }

	std::cout << std::endl << std::endl;
	return 0;
}

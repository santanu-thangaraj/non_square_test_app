
#include <iostream>
#include <CL/cl.h>
#include <algorithm>
#include "test_app.h"

#define AVAIL_MEM_SIZE (32768)

#define SWAP_WZ_MULT 350
#define GPU_VERIFY 0
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

   int verify_initial_trans(T* input, T* output);

   void verify_initial_trans_CPU();


   int verify_swap(T* input, T* output);

   void snake(T *ts, T *td, int num_lines_loaded, int is, int id, int pos, T* z);
   void verify_swap_CPU();

   ~test_inplace_transpose();
   int is_complex_planar;
   int start_inx;
   cl_mem input_output_buffer_device;
   T* x; //original data
   T* y; // GPU output
   T* z; // CPU output
 //  T* interm; //CPU intermediate data

   cl_mem input_output_buffer_device_R;
   cl_mem input_output_buffer_device_I;
   T* x_R;
   T* x_I;
   T* y_R;
   T* y_I;
   T* z_R;
   T* z_I;
//   T* interm_R;
//   T* interm_I;
   bool   use_global_memory;
   size_t global_mem_requirement_in_bytes;
   int small_dim;
   int big_dim;

private:

    int L;
    int M;

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

    int num_lines_loaded = 1;
    int num_reduced_row; 
    int num_reduced_col;

    if (L == small_dim)
    {
        num_reduced_row = std::ceil((float)small_dim / (float)(num_lines_loaded));
        num_reduced_col = big_dim / small_dim;
    }
    else
    {
        num_reduced_row = big_dim / small_dim;
        num_reduced_col = std::ceil((float)small_dim / (float)(num_lines_loaded)); 
    }

    int i;
    //memcpy(interm,z,L*M*sizeof(T));
    int *cycle_map = new int[num_reduced_row * num_reduced_col * 2];
    /* The memory required by cycle_map canniot exceed 2 times row*col by design*/

    get_cycles(cycle_map, num_reduced_row, num_reduced_col);

    T *ta = te;
    T *tb = to;
    T *ttmp;

    int inx = 0, start_index;
    for (i = 0; i < cycle_map[0]; i++)
    {
        start_index = cycle_map[++inx];
       // std::cout << "\nCycle:" << (i + 1) << ">\t" << "(" << start_index << "," << cycle_map[inx + 1] << ")";
        snake(ta, tb, num_lines_loaded, start_index, cycle_map[inx + 1], 0, z);

        while (start_index != cycle_map[++inx])
        {
            ttmp = ta;
            ta = tb;
            tb = ttmp;

      //      std::cout << "\t" << "(" << cycle_map[inx] << "," << cycle_map[inx + 1] << ")";
            int action_var = (cycle_map[inx + 1] == start_index) ? 2 : 1;
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

                for (int i = 0; i < big_dim / small_dim; i++)
                {

                    C* ptr_tmp_input = (C*)(&input[l + w * width + i * start_inx]);
                    C* ptr_tmp_output = (C*)(&output[l * width + w + i * start_inx]);

                    if (ptr_tmp_input[comp] != ptr_tmp_output[comp])
                    {
                        std::cout << "Fail\n";
                        return 1;
                    }
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
    big_dim = (L > M) ? L : M;
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



    if (!is_complex_planar)
    {
        x = new T[L*M];
        y = new T[L*M];
        z = new T[L*M];
   //     interm = new T[L*M];
    }
    else
    {
        x_R = new T[L*M];
        x_I = new T[L*M];
        y_R = new T[L*M];
        y_I = new T[L*M];
        z_R = new T[L*M];
        z_I = new T[L*M];
  //      interm_R = new T[L*M];
  //      interm_I = new T[L*M];
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
        memcpy(z, x, L*M*sizeof(T));
        memcpy(y, x, L*M*sizeof(T));
    }
    else
    {
        memcpy(z_R, x_R, L*M*sizeof(T));
        memcpy(z_I, x_I, L*M*sizeof(T));
        memcpy(y_R, x_R, L*M*sizeof(T));
        memcpy(y_I, x_I, L*M*sizeof(T));
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
  //      delete[] interm;

    }
    else
    {
        delete[] x_R;
        delete[] y_R;
        delete[] z_R;
 //       delete[] interm_R;
        delete[] x_I;
        delete[] y_I;
        delete[] z_I;
  //      delete[] interm_I;

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

template<typename T, typename C>
int master_test_function(char** argv, cl_context context, cl_command_queue commands, cl_kernel kernel_ST, cl_kernel kernel_swap)
{
    const int L = atoi(argv[3]);
    const int M = atoi(argv[4]);
    cl_double           NDRangePureExecTimeMs;

    test_inplace_transpose<T, C> test_inplace_transpose(argv, context, L, M);

    if (!test_inplace_transpose.is_complex_planar)
    {
        for (int i = 0; i < test_inplace_transpose.big_dim / test_inplace_transpose.small_dim; i++)
        { 
            test_inplace_transpose.inplace_transpose_square_CPU(i * test_inplace_transpose.start_inx, test_inplace_transpose.z);
        }
    }
    else
    {
        for (int i = 0; i < test_inplace_transpose.big_dim / test_inplace_transpose.small_dim; i++)
        {
            test_inplace_transpose.inplace_transpose_square_CPU(i * test_inplace_transpose.start_inx, test_inplace_transpose.z_R);
            test_inplace_transpose.inplace_transpose_square_CPU(i * test_inplace_transpose.start_inx, test_inplace_transpose.z_I);
        }
    }

    test_inplace_transpose.verify_initial_trans_CPU();

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
    
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    int status;

    cl_context          context = NULL;
    cl_kernel           kernel_ST = NULL;
    cl_kernel           kernel_swap = NULL;
    cl_command_queue    commands = NULL;

    enum INPUT_OUTPUT_TYPE_T input_output_type;

    input_output_type = COMPLEX_TO_COMPLEX;

    if (argc != 5)
    {
        std::cout << "The executable expects 4 arguments:\n";
        std::cout << "argument 1: R2R/C2C/P2P\n";
        std::cout << "argument 2: float/double\n";
        std::cout << "argument 3 & 4: dimensions of matrix\n";
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "R2R") == 0)
    {
        input_output_type = REAL_TO_REAL;
    }

    if (strcmp(argv[1], "P2P") == 0)
    {
        input_output_type = PLANAR_TO_PLANAR;
    }

    if (input_output_type == COMPLEX_TO_COMPLEX)
    {
        if (strcmp(argv[2], "float") == 0)
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
        if (strcmp(argv[2], "float") == 0)
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

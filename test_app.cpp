
#include <iostream>
#include <CL/cl.h>
#include "test_app.h"

#define AVAIL_MEM_SIZE 32768
#define AVAIL_MEM (AVAIL_MEM_SIZE / 4)

template<typename T, typename C>
class test_inplace_transpose
{
public:
    test_inplace_transpose(char** argv,
                           cl_context context,
                           int L,
                           int M);

   void inplace_transpose_square_CPU(int start_x);

   int inplace_transpose_square_GPU( char** argv,
                                      cl_command_queue  commands,
                                      cl_kernel kernel,
                                      cl_double *ptr_NDRangePureExecTimeMs
                                     );
   int verify_initial_trans(T* input, T* output);

   void verify_initial_trans_CPU();
   void verify_initial_trans_GPU();

   ~test_inplace_transpose();

   int start_inx;
private:
    cl_mem input_output_buffer_device;
    T* x;
    T* y;
    T* z;
    int L;
    int M;
    int small_dim;
    int width;
    
};

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_initial_trans_CPU()
{
    this->verify_initial_trans(x,z);
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::verify_initial_trans_GPU()
{
    this->verify_initial_trans(x,y);
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
    int tot_num_work_items = num_work_items_in_group * atoi(argv[6]);
    cl_event warmup_event;
    cl_event perf_event[NUM_PERF_ITERATIONS];
    /*size_t global_work_offset[MAX_DIMS] = {0,0,0};*/
    cl_ulong start = 0, end = 0;
    int i;
    cl_double perf_nums[NUM_PERF_ITERATIONS];

    global_work_size[0] = tot_num_work_items;
    local_work_size[0] = num_work_items_in_group;

    status = clSetKernelArg(kernel,
        0,
        sizeof(cl_mem),
        &input_output_buffer_device);

    if (status != CL_SUCCESS)
    {
        std::cout << "The kernel set argument failure\n";
        return EXIT_FAILURE;
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
    width = M;

    start_inx = (small_dim == L) ? L : M*M;

    x = new T[L*M];
    y = new T[L*M];
    z = new T[L*M];


    for (int l = 0; l<L; l++)
        for (int m = 0; m<M; m++)
        {
            C* ptr_tmp_x = (C*)(&x[l*M + m]);
            ptr_tmp_x[0] = rand() % 1000;
            if ((sizeof(T) / sizeof(C)) == 2)
            {
                ptr_tmp_x[1] = ptr_tmp_x[0];
            }
        }

    input_output_buffer_device = clCreateBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        L * M * sizeof(T),
        x,
        &status);

    if (status != CL_SUCCESS)
    {
        std::cout << "ERROR! Allocation of GPU input buffer failed.\n";
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

    memcpy(z, x, L*M*sizeof(T));
}

template<typename T, typename C>
test_inplace_transpose<T, C>::~test_inplace_transpose()
{
    delete[] x;
    delete[] y;
    delete[] z;
}

template<typename T, typename C>
void test_inplace_transpose<T, C>::inplace_transpose_square_CPU(int start_x)
{
	for(int l=0; l<small_dim; l++)
		for(int m=0; m<small_dim; m++)
		{
			if(m<l)
				continue;

			T tmp = z[l*width + m + start_x];
			z[l*width + m + start_x] = z[m*width + l + start_x];
			z[m*width + l + start_x] = tmp;
		}
}

void snake(int *x, int *ts, int *td, const int &L, const int &P, int is, int id, int pos)
{
	// pos 0 - start, 1 - inter, 2 - end

	for(int p=0; p<P; p++)
	{
		for(int j=0; j<L; j++)
		{
			if(pos == 0)
			{
				ts[p*L + j] = x[is*P*L + p*L + j];
				td[p*L + j] = x[id*P*L + p*L + j];				
				x[id*P*L + p*L + j] = ts[p*L + j];		
			}
			else if(pos == 1)
			{
				td[p*L + j] = x[id*P*L + p*L + j];				
				x[id*P*L + p*L + j] = ts[p*L + j];		
			}
			else
			{
				x[id*P*L + p*L + j] = ts[p*L + j];
			}
		}
	}
}
/* -> get_cycles funcition gets the swapping logic required for given row x col matrix.
   -> cycle_map[0] holds the total number of cycles required. 
   -> cycles start and end with the same index, hence we can identify individual cycles,
   though we tend to store the cycle index contiguousl.y*/
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
    if ( L < max_capacity)
    {
        return L;
    }

    int square_root = (int)sqrt(L) + 1;
    int max_factor = 1;
    for (int i = 1; i < square_root; i++)
    {
        if (L % i == 0)
        {
            if (( i > max_factor) && (i <= max_capacity))
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

void inplace_1_isto_2_transpose_generic(int *x, const int &L, const int &M)
{
    int *te = new int[AVAIL_MEM >> 1];
    int *to = new int[AVAIL_MEM >> 1];
    int max_capacity = (AVAIL_MEM >> 1) / L;
    if (max_capacity <= 0)
    {
        std::cout << "\nIn-place transpose cannot be performed within specified memory constraints.\n";
        exit(1);
    }
    int num_lines_loaded = get_num_lines_to_be_loaded(max_capacity, L);
    int num_reduced_row = std::ceil( (float) M / (float) (2 * num_lines_loaded)); 
    int num_reduced_col = 2;
    /* The reduced row and col comes from the fact that we perform swaps of 'num_lines_loaded' rows first
       and thus reducing the amount of swaps required to be done using the snake function*/
    int i;

    if (num_lines_loaded > 1)
    {
        for (i = 0; i < M; i += 2 * num_lines_loaded)
        {
            // read
            for (int p = 0; p < num_lines_loaded; p++)
            {
                for (int j = 0; j < L; j++)
                {
                    te[p*L + j] = x[i*L + (2 * p + 0)*L + j];
                    to[p*L + j] = x[i*L + (2 * p + 1)*L + j];
                }
            }

            // write
            for (int p = 0; p < num_lines_loaded; p++)
            {
                for (int j = 0; j < L; j++)
                {
                    x[i*L + 0 + p*L + j] = te[p*L + j];
                    x[i*L + num_lines_loaded*L + p*L + j] = to[p*L + j];
                }
            }
        }
    }
    int *cycle_map = new int[num_reduced_row * num_reduced_col * 2];
    /* The memory required by cycle_map canniot exceed 2 times row*col by design*/

    get_cycles(cycle_map, num_reduced_row, num_reduced_col);

    int *ta = te;
    int *tb = to;
    int *ttmp;

    int inx = 0, start_inx;
    for (i = 0; i < cycle_map[0]; i++)
    {
        start_inx = cycle_map[++inx];
        std::cout << "\nCycle:" << (i + 1) <<">\t"<< "("<<start_inx <<","<< cycle_map[inx + 1] <<")";
        snake(x, ta, tb, L, num_lines_loaded, start_inx, cycle_map[inx + 1], 0);

        while (start_inx != cycle_map[++inx])
        {
            ttmp = ta;
            ta = tb;
            tb = ttmp;

            std::cout <<"\t" << "(" <<cycle_map[inx] << "," << cycle_map[inx + 1] << ")";
            int action_var = (cycle_map[inx + 1] == start_inx) ? 2 : 1;
            snake(x, ta, tb, L, num_lines_loaded, cycle_map[inx], cycle_map[inx + 1], action_var);
        }
    }

    delete[] te;
    delete[] to;
}

int opencl_build_kernel(char** argv,
    cl_context* ptr_context,
    cl_command_queue* ptr_commands,
    cl_kernel* ptr_kernel,
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

    *ptr_kernel = clCreateKernel(program, "transpose_nonsquare", &err);
    if (!(*ptr_kernel) || err != CL_SUCCESS)
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
int opencl_allocate_buffer(char** argv,
    cl_context context,
    cl_mem* ptr_input_output_buffer_device,
    const int L,
    const int M)
{
    cl_int status;

    x = new T[L*M];
    y = new T[L*M];
    z = new T[L*M];

    *ptr_input_output_buffer_device = clCreateBuffer(context,
        CL_MEM_READ_WRITE,
        L * M * sizeof(T),
        NULL,
        &status);

    if (status != CL_SUCCESS)
    {
        std::cout << "ERROR! Allocation of GPU input buffer failed.\n";
        return EXIT_FAILURE;
    }

    for (int l = 0; l<L; l++)
        for (int m = 0; m<M; m++)
        { 
            C* ptr_tmp_x = (C*)(&x[l*M + m]);
            ptr_tmp_x[0] = (C)rand();
            if ((sizeof(T) / sizeof(C)) == 2)
            {
                ptr_tmp_x[1] = ptr_tmp_x[0];
            }
        }


    for (int l = 0; l<L; l++)
        for (int m = 0; m<M; m++)
        {
            C* ptr_tmp_x = (C*)(&x[l*M + m]);
            C* ptr_tmp_y = (C*)(&y[m*L + l]);
            ptr_tmp_y[0] = ptr_tmp_x[0];
            if ((sizeof(T) / sizeof(C)) == 2)
            {
                ptr_tmp_y[1] = ptr_tmp_x[1];
            }
        }

    memcpy(z, x, L*M*sizeof(T));
}

int main(int argc, char** argv)
{
	const int L = atoi(argv[4]);
	const int M = atoi(argv[5]);

    int status;

    cl_context          context;
    cl_kernel           kernel;
    cl_command_queue    commands;

    cl_mem              input_output_buffer_device;
    cl_double           NDRangePureExecTimeMs;

    enum INPUT_OUTPUT_TYPE_T input_output_type;

    input_output_type = COMPLEX_TO_COMPLEX;

    if (strcmp(argv[1], "R2R") == 0)
    {
        input_output_type = REAL_TO_REAL;
    }

    /*initialize the command queues and build kernel*/
    status = opencl_build_kernel(argv,
        &context,
        &commands,
        &kernel,
        input_output_type);

    if (status == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }

    if (input_output_type == COMPLEX_TO_COMPLEX)
    {
        if (strcmp(argv[3], "float") == 0)
        {
            test_inplace_transpose<cl_float2, cl_float> test_inplace_transpose(argv, context, L, M);

            test_inplace_transpose.inplace_transpose_square_CPU(0);
            test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx);

            test_inplace_transpose.inplace_transpose_square_GPU(argv, commands, kernel, &NDRangePureExecTimeMs);

            test_inplace_transpose.verify_initial_trans_CPU();
            test_inplace_transpose.verify_initial_trans_GPU();
        }
        else
        {
            test_inplace_transpose<cl_double2, cl_double> test_inplace_transpose(argv, context, L, M);

            test_inplace_transpose.inplace_transpose_square_CPU(0);
            test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx);

            test_inplace_transpose.inplace_transpose_square_GPU(argv, commands, kernel, &NDRangePureExecTimeMs);

            test_inplace_transpose.verify_initial_trans_CPU();
            test_inplace_transpose.verify_initial_trans_GPU();
        }
    }
    else
    {
        if (strcmp(argv[3], "float") == 0)
        {
            test_inplace_transpose<cl_float, cl_float> test_inplace_transpose(argv, context, L, M);

            test_inplace_transpose.inplace_transpose_square_CPU(0);
            test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx);

            test_inplace_transpose.inplace_transpose_square_GPU(argv, commands, kernel, &NDRangePureExecTimeMs);

            test_inplace_transpose.verify_initial_trans_CPU();
            test_inplace_transpose.verify_initial_trans_GPU();

        }
        else
        {
            test_inplace_transpose<cl_double, cl_double> test_inplace_transpose(argv, context, L, M);

            test_inplace_transpose.inplace_transpose_square_CPU(0);
            test_inplace_transpose.inplace_transpose_square_CPU(test_inplace_transpose.start_inx);

            test_inplace_transpose.inplace_transpose_square_GPU(argv, commands, kernel, &NDRangePureExecTimeMs);

            test_inplace_transpose.verify_initial_trans_CPU();
            test_inplace_transpose.verify_initial_trans_GPU();

        }
    }



	// for(int i=0; i<L*M; i++)
	// {
	// 	if(!(i%M))
	// 		std::cout << std::endl;
	// 
	// 	std::cout.width(4);
	// 	std::cout << z[i] << " ";
	// }
	// std::cout << std::endl << std::endl;

	//inplace_transpose_square_CPU(z, L, M);

	// for(int i=0; i<L*M; i++)
	// {
	// 	if(!(i%M))
	// 		std::cout << std::endl;
	// 
	// 	std::cout.width(4);
	// 	std::cout << z[i] << " ";
	// }
	// std::cout << std::endl << std::endl;

	//inplace_transpose_square_CPU(z+L, L, M);

	// for(int i=0; i<L*M; i++)
	// {
	// 	if(!(i%M))
	// 		std::cout << std::endl;
	// 
	// 	std::cout.width(4);
	// 	std::cout << z[i] << " ";
	// }
	// std::cout << std::endl << std::endl;

 


	std::cout << std::endl << std::endl;
	return 0;
}

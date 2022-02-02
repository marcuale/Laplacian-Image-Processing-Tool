/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2020 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#include <stdio.h>
#include <string>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "pgm.h"
#include "kernels.h"

/***** REDUCTION FUNCTION ******/
/** Reduction function is similar for some kernels, thus we keep it here **/
__global__ void reduction(int32_t *image, int32_t image_size, int32_t *largest,
		int32_t *smallest){
	
	__shared__ int min_cache[512];			// Reduction will be done with 512 threads per block (experimentation needed)
	__shared__ int max_cache[512];			// Need a cache to store both potential max and potential mins
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int cache_index = threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	
	int min_temp = INT_MAX;
	int max_temp = INT_MIN;
	
	while(index < image_size){
		if(image[index] < min_temp) min_temp = image[index];		// check for new min
		if(image[index] > max_temp) max_temp = image[index];		// check for new max
		
		index += stride;		// Thread continues iteration through array
	}
	
	// Threads set their max/min candidates
	min_cache[cache_index] = min_temp;
	max_cache[cache_index] = max_temp;	
	
	__syncthreads();		// All threads need to store their potential max/mins before we can proceed
		
	int reduction_stride = blockDim.x / 2;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	// Perform reduction
	while(reduction_stride != 0){
		
		// reduction for min value
		if(index < image_size){
			if(cache_index < reduction_stride && min_cache[cache_index] > min_cache[cache_index + reduction_stride]){
				min_cache[cache_index] = min_cache[cache_index + reduction_stride];
			}
			// reduction for max value
			if(cache_index < reduction_stride && max_cache[cache_index] < max_cache[cache_index + reduction_stride]){
				max_cache[cache_index] = max_cache[cache_index + reduction_stride];
			}
		}
		
		__syncthreads();
		
		reduction_stride /= 2;
	}
	
	// Update global max/min
	if(threadIdx.x == 0){		// first thread of every block responsible updating global max/min
		//while(atomicCAS(mutex, 0, 1) != 0);		// basic spin lock
		//*smallest = min(*smallest, min_cache[0]);			// reduction technique stores the block's min/max at index 0
		//*largest = max(*largest, max_cache[0]);
		atomicMin(smallest, min_cache[0]);
		atomicMax(largest, max_cache[0]);
		//results_max[blockIdx.x] = max_cache[0];
		//results_min[blockIdx.x] = min_cache[0];
	}
		
}


/************** FILTER CONSTANTS*****************/
/** DESCRIBED IN HAND OUT, BUT TO REITERATE, WHEN 
INVOKING KERNEL CALLS, YOU MUST SPECIFY DIMENSION MANUALLY **/

/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
	
	
int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };	
	
/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
	
/* Identity */
int8_t identity_m[] = {1};

/************** CPU Implementation Data Structure Definitions **************/
/* Common attributes between threads */
typedef struct common_work_t
{
    int8_t *f;
	int dimension;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
	int32_t chunk_size;
    pthread_barrier_t barrier;
} common_work;

/* Each thread assigned a work_t struct to pass arguments to respective functions */
typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;

/************** Shared Global Data For CPU Implementation*****************/

/* Synchronization primitives */
pthread_mutex_t write_mutex;

/* Maximum and Minimum processed pixel values */
int global_max;
int global_min;

/*************** COMMON WORK For CPU Implementation***********************/

/* Process a single pixel and returns the value of processed pixel* */
int32_t apply2d(int8_t *f, int dimension, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int mid = dimension/2;
    int i, j, cur_row, cur_col, og_pos, filter_pos;
    int32_t result = 0;
    for (i=-mid; i<=mid; i++){
        for (j=-mid; j<=mid; j++){
            cur_row = row + j;
            cur_col = column + i;
            if ((cur_row>=0) && (cur_col>=0) && (cur_row<height) && (cur_col<width)){
                og_pos = width*cur_row + cur_col;
                filter_pos = dimension*(i+mid) +(j+mid);
                result += original[og_pos] * f[filter_pos];
            }
        }
    }
    target[width*row + column] = result;
    return result;
}

/* Modifies the global maximum and minimum values of processed pixels synchronously */
void modify_global_max_min(int32_t local_max, int32_t local_min){
    pthread_mutex_lock(&write_mutex);
    if (local_max > global_max){
        global_max = local_max;
    }
    if (local_min < global_min){
        global_min = local_min;
    }
    pthread_mutex_unlock(&write_mutex);
}

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, 
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }
    
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* SHARDED_COLUMNS_ROW_MAJOR */
/* Thread entry point for SHARDED_COLUMNS_ROW_MAJOR method */
void* sharded_columns_row_major(void *task)    
{    
    // get work attributes
    work *task_work = (work *) task;
    common_work *common = task_work->common;
    int id = task_work->id;

    // get common attributes
    int8_t *f = common->f;
	int dimension = common->dimension;
    const int32_t *original = common->original_image;
    int32_t *target = common->output_image;
    int32_t width = common->width;
    int32_t height = common->height;
    int num_threads = common->max_threads;

    // get the range of columns the worker thread need to go through
    int num_columns = width / num_threads; // number of columns for each thread except for the last one.
    int start_col = id * num_columns; // the starting columns the worker thread will work on
    int end_col = (id * num_columns) + num_columns; // the ending thread the worker thread will work on
    if ((width % num_threads) != 0 && (id+1 == num_threads)) { // the last work thread
        end_col = width;
    }

    // row major here, reverse order for column major
    int32_t i, j, result, smallest, largest;
    smallest = INT_MAX;
    largest = INT_MIN;
    for (i=0; i<height; i++){
        for (j=start_col; j<end_col; j++){
            result = apply2d(f, dimension, original, target, width, height, i, j);
            if (result > largest){ 
                largest = result;
            }
            if (result < smallest){
                smallest = result;
            }
        }
    }

    // wait for other threads
    pthread_barrier_wait(&(common->barrier));
    // add local maximum and minimum data to the global maximum and minimum
    modify_global_max_min(largest, smallest);

    return NULL;
}

// Get time elapsed between t0 and t1  
static inline struct timespec difftimespec(struct timespec t1, struct timespec t0)  
{  
    assert(t1.tv_nsec < 1000000000);  
    assert(t0.tv_nsec < 1000000000);  
  
    return (t1.tv_nsec >= t0.tv_nsec)  
        ? (struct timespec){ t1.tv_sec - t0.tv_sec    , t1.tv_nsec - t0.tv_nsec             }  
        : (struct timespec){ t1.tv_sec - t0.tv_sec - 1, t1.tv_nsec - t0.tv_nsec + 1000000000};  
}  

// Convert struct timespec to milliseconds  
static inline double timespec_to_msec(struct timespec t)  
{  
    return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;  
}

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
        float time_gpu_transfer_in, float time_gpu_transfer_out)
{
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu/time_gpu_computation);
    printf("%7.2f\n", time_cpu/
            (time_gpu_computation  + time_gpu_transfer_in + time_gpu_transfer_out));
}

int main(int argc, char **argv)
{
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3)
    {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt (argc, argv, "i:o:")) != -1)
    {
        switch (c)
        {
            case 'i':
                input_filename = std::string(optarg);
                break;
            case 'o':
                cpu_output_filename = std::string(optarg);
                base_gpu_output_filename = std::string(optarg);
                break;
            default:
                return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
    {
       printf("Error loading source image.\n");
       return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
            "Speedup_noTrf Speedup\n");

    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img); 

    //float time_cpu;
    /* TODO: run your CPU implementation here and get its time. Don't include
     * file IO in your measurement.*/
   // save_pgm_to_file(cpu_output_filename.c_str(), &cpu_output_img);


    /* TODO:
     * run each of your gpu implementations here,
     * get their time,
     * and save the output image to a file.
     * Don't forget to add the number of the kernel
     * as a prefix to the output filename:
     * Print the execution times by calling print_run().
     */
	 
	 
	 
	 
	 /** SET UP IMAGE **/
	
	 // Set up image to test
	 //pgm_image h_kernel1_test_img;
	 int width = source_img.width;						// To test different size images, change these 2 values
	 int height = source_img.height;
	 int image_size = width * height;
	 //create_random_pgm_image(&h_kernel1_test_img, width, height);
	 //copy_pgm_image_size(&h_kernel1_test_img, &cpu_output_img);
	
	 /** VARIABLES/WORK COMMON TO ALL KERNELS **/
	 
	 float time_gpu_transfer_in, time_gpu_computation_time, time_gpu_transfer_out; // timing values

	 // Shared (repeated) kernel variables
	 int num_threads = 0;
	 int num_blocks = 0;

	 // Vars needed to find global max/min
	 int *h_largest = (int *) malloc(sizeof(int));
	 int *d_largest;
	 int *h_smallest = (int *) malloc(sizeof(int));
	 int *d_smallest;
	 //int *d_mutex;
	 
	 // Set up filter
	 int8_t *h_filter = lp3_m;		// can change to different filter (hard coded above)
	 int filter_dimension = 3;				// remember to change dimension if filter is changed
	 int filter_size = filter_dimension * filter_dimension;
	 
	 // Allocate and transfer data to GPU
	 int8_t *d_filter;
	 
	 int32_t *h_input_matrix;
	 int32_t *d_input_matrix;
	 
	 int32_t *h_output_matrix = (int32_t*) malloc(image_size * sizeof(int32_t));
	 int32_t *d_output_matrix;
	 
	 h_input_matrix = source_img.matrix;
	
	 
	 
	/*** CPU Implementation ***/
	num_threads = 4;
	global_max = INT32_MIN;
    global_min = INT32_MAX;

	// Initialize shared thread primitives
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);
	pthread_mutex_init(&write_mutex, NULL);
	
	// Gather and set all common attributes between threads
    common_work *shared_work = (common_work *)malloc(sizeof(common_work));
    shared_work->f = h_filter;
	shared_work->dimension = filter_dimension;
    shared_work->original_image = h_input_matrix;
    shared_work->output_image = h_output_matrix;
    shared_work->width = width;
    shared_work->height = height;
    shared_work->max_threads = num_threads;
    shared_work->barrier = barrier;
	shared_work->chunk_size = 0;

    pthread_t workers[num_threads];
    // initialize work (struct) here
    work *task;
	
	struct timespec start_cpu, stop_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
	
	// Create threads
	int i;
    for (i=0; i<num_threads; i++){

        // fulfill each task
        task = (work *) malloc(sizeof(work));
        task->common = shared_work;
        task->id = i;

		pthread_create(&workers[i], NULL, sharded_columns_row_major, (void *)task);  
		
    }
	
	// join
    int j;
    for (j = 0; j < num_threads; j++) {
        if(pthread_join(workers[j], NULL) != 0){
            perror("Failed to join threads");
        }
    }
	
	// normalize with global max and min
    int size = width*height;
    for (i=0; i<size; i++){
        normalize_pixel(shared_work->output_image, i, global_min, global_max);
    }
	
	clock_gettime(CLOCK_MONOTONIC, &stop_cpu);
	
	double time_cpu = timespec_to_msec(difftimespec(stop_cpu, start_cpu));
	
	

    // finish barrier here
    pthread_barrier_destroy(&barrier);
	
	cpu_output_img.matrix = shared_work->output_image;
	save_pgm_to_file(cpu_output_filename.c_str(), &cpu_output_img);
	
	
	/** Start GPU Implementations **/
	 
	 
	 // Start Time gpu transfer_in
	 cudaEvent_t start, stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 // Image input/output allocation
	 cudaMalloc((void**)&d_filter, filter_size * sizeof(int8_t));
	 cudaMalloc((void**)&d_input_matrix, image_size * sizeof(int32_t));
	 cudaMalloc((void**)&d_output_matrix, image_size * sizeof(int32_t));
	 
	 cudaMemcpy(d_filter, h_filter, filter_size * sizeof(int8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_input_matrix, h_input_matrix, image_size * sizeof(int32_t), cudaMemcpyHostToDevice);
	 
	 // Global max/min allocation
	 cudaMalloc((void**)&d_largest, sizeof(int));
	 cudaMalloc((void**)&d_smallest, sizeof(int));
	 //cudaMalloc((void**)&d_mutex, sizeof(int));
	 
	 cudaMemset(d_largest, INT_MIN, sizeof(int));
	 cudaMemset(d_smallest, INT_MAX, sizeof(int));
	 //cudaMemset(d_mutex, 0, sizeof(int));
	 
	 // End Time gpu transfer_in
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_in, start, stop);
	 
	 /********* KERNEL 1 *********/
{
	 num_threads = image_size;		// One thread per pixel for Kernel 1 and 2
	 
	 // If the image is smaller than our max thread per block (1024 on these gpus) 
	 // we invoke the kernel with just number of threads
	 // else we calculate how many blocks we need
	 
	 // Start time gpu_computation
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 if(image_size <= 1024){
		 //invoke filter kernel just using threads
		 num_blocks = 1;
		 kernel1<<<num_blocks, image_size>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);		// also this one in general terms
	 } else {
		num_blocks = (num_threads % 512 == 0) ? num_threads / 512 : (num_threads / 512) + 1;
		//invoke filter kernel 
		kernel1<<<num_blocks, 512>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);
	 }
	 
	 cudaDeviceSynchronize();
	  
	 reduction<<<num_blocks, 512>>>(d_output_matrix, image_size, d_largest, d_smallest);
	 
	 // Copy max and min back to host to check values (but when we apply normalization we don't really have to, save time that way
	 cudaMemcpy(h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);
	 cudaMemcpy(h_smallest, d_smallest, sizeof(int), cudaMemcpyDeviceToHost);
	 
	 if(num_blocks == 1){
		normalize1<<<num_blocks, image_size>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 } else {
		normalize1<<<num_blocks, 512>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 }
	 
	 // End time gpu computation 
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_computation_time, start, stop);
	 
	 // Transfer data back
	 
	 // Start Time transfer out
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 cudaMemcpy(h_output_matrix, d_output_matrix, image_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
	 
	 // End time transfer out
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_out, start, stop);
	
	 print_run(time_cpu, 1, time_gpu_computation_time, time_gpu_transfer_in, time_gpu_transfer_out);
	 
	 //save file
	 std::string gpu_file1 = "1"+base_gpu_output_filename;
	 cpu_output_img.matrix = h_output_matrix;
	 save_pgm_to_file(gpu_file1.c_str(), &cpu_output_img);

	 cudaDeviceSynchronize(); 
	 
	 cpu_output_img.matrix = h_output_matrix;


	 cudaFree(d_filter);
	 cudaFree(d_input_matrix);
	 cudaFree(d_output_matrix);
	 cudaFree(d_largest);
	 cudaFree(d_smallest);
}
	 
	 /********* KERNEL 2 *********/
{
	 // Start Time gpu transfer_in
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 // Image input/output allocation
	 cudaMalloc((void**)&d_filter, filter_size * sizeof(int8_t));
	 cudaMalloc((void**)&d_input_matrix, image_size * sizeof(int32_t));
	 cudaMalloc((void**)&d_output_matrix, image_size * sizeof(int32_t));
	 
	 cudaMemcpy(d_filter, h_filter, filter_size * sizeof(int8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_input_matrix, h_input_matrix, image_size * sizeof(int32_t), cudaMemcpyHostToDevice);
	 
	 // Global max/min allocation
	 cudaMalloc((void**)&d_largest, sizeof(int));
	 cudaMalloc((void**)&d_smallest, sizeof(int));
	 //cudaMalloc((void**)&d_mutex, sizeof(int));
	 
	 cudaMemset(d_largest, INT_MIN, sizeof(int));
	 cudaMemset(d_smallest, INT_MAX, sizeof(int));
	 //cudaMemset(d_mutex, 0, sizeof(int));
	 
	 // End Time gpu transfer_in
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_in, start, stop);
	 
	 num_threads = image_size;		// One thread per pixel for Kernel 1 and 2
	 
	 // If the image is smaller than our max thread per block (1024 on these gpus) 
	 // we invoke the kernel with just number of threads
	 // else we calculate how many blocks we need
	 
	 // Start time gpu_computation
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 if(image_size <= 1024){
		 //invoke filter kernel just using threads
		 num_blocks = 1;
		 kernel2<<<num_blocks, image_size>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);		// also this one in general terms
	 } else {
		 num_blocks = (num_threads % 512 == 0) ? num_threads / 512 : (num_threads / 512) + 1;
		 //invoke filter kernel 
		 kernel2<<<num_blocks, 512>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);
	 }
	 
	 cudaDeviceSynchronize();
	 
	 // Reduction to find global max/min
	 reduction<<<num_blocks, 512>>>(d_output_matrix, image_size, d_largest, d_smallest);
	 
	 // Copy max and min back to host to check values (but when we apply normalization we don't really have to, save time that way
	 cudaMemcpy(h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);
	 cudaMemcpy(h_smallest, d_smallest, sizeof(int), cudaMemcpyDeviceToHost);
	 
	 if(num_blocks == 1){
		normalize2<<<num_blocks, image_size>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 } else {
		normalize2<<<num_blocks, 512>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 }
	 
	 // End time gpu computation 
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_computation_time, start, stop);
	 
	 // Transfer data back
	 
	 // Start Time transfer out
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 cudaMemcpy(h_output_matrix, d_output_matrix, image_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
	 
	 // End time transfer out
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_out, start, stop);
	 

	 print_run(time_cpu, 2, time_gpu_computation_time, time_gpu_transfer_in, time_gpu_transfer_out);
	 
	 // save file
	 std::string gpu_file2 = "2"+base_gpu_output_filename;
	 cpu_output_img.matrix = h_output_matrix;
	 save_pgm_to_file(gpu_file2.c_str(), &cpu_output_img);

	 cudaDeviceSynchronize();
	 
	 cudaFree(d_filter);
	 cudaFree(d_input_matrix);
	 cudaFree(d_output_matrix);
	 cudaFree(d_largest);
	 cudaFree(d_smallest);
}

	/********* KERNEL 3 *********/
{
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 // Image input/output allocation
	 cudaMalloc((void**)&d_filter, filter_size * sizeof(int8_t));
	 cudaMalloc((void**)&d_input_matrix, image_size * sizeof(int32_t));
	 cudaMalloc((void**)&d_output_matrix, image_size * sizeof(int32_t));
	 
	 cudaMemcpy(d_filter, h_filter, filter_size * sizeof(int8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_input_matrix, h_input_matrix, image_size * sizeof(int32_t), cudaMemcpyHostToDevice);
	 
	 // Global max/min allocation
	 cudaMalloc((void**)&d_largest, sizeof(int));
	 cudaMalloc((void**)&d_smallest, sizeof(int));
	 //cudaMalloc((void**)&d_mutex, sizeof(int));
	 
	 cudaMemset(d_largest, INT_MIN, sizeof(int));
	 cudaMemset(d_smallest, INT_MAX, sizeof(int));
	 //cudaMemset(d_mutex, 0, sizeof(int));
	 
	 // End Time gpu transfer_in
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_in, start, stop);
	
	 num_threads = height;		// One thread per pixel for Kernel 1 and 2
	 
	 // If the image is smaller than our max thread per block (1024 on these gpus) 
	 // we invoke the kernel with just number of threads
	 // else we calculate how many blocks we need
	 
	 // Start time gpu_computation
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 if(num_threads <= 1024){
		 //invoke filter kernel just using threads
		 num_blocks = 1;
		 kernel3<<<num_blocks, num_threads>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);		// also this one in general terms
	 } else {
		 num_blocks = (num_threads % 512 == 0) ? num_threads / 512 : (num_threads / 512) + 1;
		 //invoke filter kernel 
		 kernel3<<<num_blocks, 512>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);
	 }
	 
	 cudaDeviceSynchronize();
	 
	 // Reduction to find global max/min
	 reduction<<<num_blocks, 512>>>(d_output_matrix, image_size, d_largest, d_smallest);
	 
	 // Copy max and min back to host to check values (but when we apply normalization we don't really have to, save time that way
	 cudaMemcpy(h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);
	 cudaMemcpy(h_smallest, d_smallest, sizeof(int), cudaMemcpyDeviceToHost);
	 
	 if(num_blocks == 1){
		normalize3<<<num_blocks, num_threads>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 } else {
		normalize3<<<num_blocks, 512>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 }
	 
	 // End time gpu computation 
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_computation_time, start, stop);
	 
	 // Transfer data back
	 
	 // Start Time transfer out
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 cudaMemcpy(h_output_matrix, d_output_matrix, image_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
	 
	 // End time transfer out
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_out, start, stop);


	 print_run(time_cpu, 3, time_gpu_computation_time, time_gpu_transfer_in, time_gpu_transfer_out);

	 // save file
	 std::string gpu_file3 = "3"+base_gpu_output_filename;
	 cpu_output_img.matrix = h_output_matrix;
	 save_pgm_to_file(gpu_file3.c_str(), &cpu_output_img);
	 
	 cudaDeviceSynchronize(); 
	 
	 cudaFree(d_filter);
	 cudaFree(d_input_matrix);
	 cudaFree(d_output_matrix);
	 cudaFree(d_largest);
	 cudaFree(d_smallest);
}

	/********* KERNEL 4 *********/
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Image input/output allocation
    cudaMalloc((void**)&d_filter, filter_size * sizeof(int8_t));
    cudaMalloc((void**)&d_input_matrix, image_size * sizeof(int32_t));
    cudaMalloc((void**)&d_output_matrix, image_size * sizeof(int32_t));
    
    cudaMemcpy(d_filter, h_filter, filter_size * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_matrix, h_input_matrix, image_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // Global max/min allocation
    cudaMalloc((void**)&d_largest, sizeof(int));
    cudaMalloc((void**)&d_smallest, sizeof(int));
    //cudaMalloc((void**)&d_mutex, sizeof(int));
    
    cudaMemset(d_largest, INT_MIN, sizeof(int));
    cudaMemset(d_smallest, INT_MAX, sizeof(int));
    //cudaMemset(d_mutex, 0, sizeof(int));
    
    // End Time gpu transfer_in
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu_transfer_in, start, stop);
   
    num_threads = 512;		// kernel 4
    
    // Start time gpu_computation
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    num_blocks = (num_threads % 512 == 0) ? num_threads / 512 : (num_threads / 512) + 1;
    //invoke filter kernel 
    kernel4<<<num_blocks, 512>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);
    
    cudaDeviceSynchronize();
    
    // Reduction to find global max/min
    reduction<<<num_blocks, 512>>>(d_output_matrix, image_size, d_largest, d_smallest);
    
    // Copy max and min back to host to check values (but when we apply normalization we don't really have to, save time that way
    cudaMemcpy(h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smallest, d_smallest, sizeof(int), cudaMemcpyDeviceToHost);
    
    normalize4<<<num_blocks, 512>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
    
    // End time gpu computation 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu_computation_time, start, stop);
    
    // Transfer data back
    
    // Start Time transfer out
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    cudaMemcpy(h_output_matrix, d_output_matrix, image_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
	
    print_run(time_cpu, 4, time_gpu_computation_time, time_gpu_transfer_in, time_gpu_transfer_out);

	// save file
    std::string gpu_file4 = "4"+base_gpu_output_filename;
	cpu_output_img.matrix = h_output_matrix;
	save_pgm_to_file(gpu_file4.c_str(), &cpu_output_img);

    // End time transfer out
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu_transfer_out, start, stop);


    
    cudaDeviceSynchronize(); 
    
    cudaFree(d_filter);
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
    cudaFree(d_largest);
    cudaFree(d_smallest);
}


 
	/********* KERNEL 5 *********/
{
	 // Start Time gpu transfer_in
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 // Image input/output allocation
	 cudaMalloc((void**)&d_filter, filter_size * sizeof(int8_t));
	 cudaMalloc((void**)&d_input_matrix, image_size * sizeof(int32_t));
	 cudaMalloc((void**)&d_output_matrix, image_size * sizeof(int32_t));
	 
	 cudaMemcpy(d_filter, h_filter, filter_size * sizeof(int8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_input_matrix, h_input_matrix, image_size * sizeof(int32_t), cudaMemcpyHostToDevice);
	 
	 // Global max/min allocation
	 cudaMalloc((void**)&d_largest, sizeof(int));
	 cudaMalloc((void**)&d_smallest, sizeof(int));
	 //cudaMalloc((void**)&d_mutex, sizeof(int));
	 
	 cudaMemset(d_largest, INT_MIN, sizeof(int));
	 cudaMemset(d_smallest, INT_MAX, sizeof(int));
	 //cudaMemset(d_mutex, 0, sizeof(int));
	 
	 // End Time gpu transfer_in
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_in, start, stop);
	 
	 num_threads = image_size;		// One thread per pixel for Kernel 1 and 2
	 
	 // If the image is smaller than our max thread per block (1024 on these gpus) 
	 // we invoke the kernel with just number of threads
	 // else we calculate how many blocks we need
	 
	 // Start time gpu_computation
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 if(image_size <= 1024){
		 //invoke filter kernel just using threads
		 num_blocks = 1;
		 kernel5<<<num_blocks, image_size>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);		// also this one in general terms
	 } else {
		 num_blocks = (num_threads % 512 == 0) ? num_threads / 512 : (num_threads / 512) + 1;
		 //invoke filter kernel 
		 kernel5<<<num_blocks, 512>>>(d_filter, filter_dimension, d_input_matrix, d_output_matrix, width, height);
	 }
	 
	 cudaDeviceSynchronize();
	 
	 // Reduction to find global max/min
	 reduction<<<num_blocks, 512>>>(d_output_matrix, image_size, d_largest, d_smallest);
	 
	 // Copy max and min back to host to check values (but when we apply normalization we don't really have to, save time that way
	 cudaMemcpy(h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);
	 cudaMemcpy(h_smallest, d_smallest, sizeof(int), cudaMemcpyDeviceToHost);
	 
	 if(num_blocks == 1){
		normalize5<<<num_blocks, image_size>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 } else {
		normalize5<<<num_blocks, 512>>>(d_output_matrix, width, height, *h_smallest, *h_largest);
	 }
	 
	 // End time gpu computation 
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_computation_time, start, stop);
	 
	 // Transfer data back
	 
	 // Start Time transfer out
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start);
	 
	 cudaMemcpy(h_output_matrix, d_output_matrix, image_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
	 
	 // End time transfer out
	 cudaEventRecord(stop);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time_gpu_transfer_out, start, stop);
	 

	 print_run(time_cpu, 5, time_gpu_computation_time, time_gpu_transfer_in, time_gpu_transfer_out);
	 
	 // save file
	 std::string gpu_file5 = "5"+base_gpu_output_filename;
	 cpu_output_img.matrix = h_output_matrix;
	 save_pgm_to_file(gpu_file5.c_str(), &cpu_output_img);

	 cudaDeviceSynchronize();
	 
	 cudaFree(d_filter);
	 cudaFree(d_input_matrix);
	 cudaFree(d_output_matrix);
	 cudaFree(d_largest);
	 cudaFree(d_smallest);
}

	 
    /* For example: */
    //std::string gpu_file1 = "1"+base_gpu_output_filename;
    //std::string gpu_file2 = "2"+base_gpu_output_filename;
    //std::string gpu_file3 = "3"+base_gpu_output_filename;
    //std::string gpu_file4 = "4"+base_gpu_output_filename;
    //std::string gpu_file5 = "5"+base_gpu_output_filename;

    //pgm_image gpu_output_img1;
    //copy_pgm_image_size(&source_img, &gpu_output_img1);
    //my_kernel_1(args...);
    //print_run(args...)
    //save_pgm_to_file(gpu_file1.c_str(), &gpu_output_img1);

    /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
     * understand the idea. */
    //std::string gpu_file2 = "2"+base_gpu_output_filename;
    //std::string gpu_file3 = "3"+base_gpu_output_filename;
    //std::string gpu_file4 = "4"+base_gpu_output_filename;
    //std::string gpu_file5 = "5"+base_gpu_output_filename;
}
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

#include "filters.h"
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
filter lp5_f = {5, lp5_m};

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
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};


/************** Data structure definitions *****************/

/* --- Structs used for parallelization --- */

/* Common attributes between threads */
typedef struct common_work_t
{
    const filter *f;
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

/* --- Queue ADT implementation struct --- */
typedef struct queue_t {
	struct queue_t *next;
	int32_t start_index;
	int32_t x_chunk;
	int32_t y_chunk;
} chunk_queue;

/************** Shared Global Data *****************/

/* Synchronization primitives */
pthread_mutex_t write_mutex;
pthread_mutex_t queue_mutex;

/* Maximum and Minimum processed pixel values */
int global_max;
int global_min;

/* Queue ADT implementation, reference to front and back for O(1) insertion and deletion */
chunk_queue *q_front = NULL;
chunk_queue *q_back = NULL;


/*************** COMMON WORK ***********************/

/* Process a single pixel and returns the value of processed pixel* */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int mid = (f->dimension)/2;
    int i, j, cur_row, cur_col, og_pos, filter_pos;
    int32_t result = 0;
    for (i=-mid; i<=mid; i++){
        for (j=-mid; j<=mid; j++){
            cur_row = row + j;
            cur_col = column + i;
            if ((cur_row>=0) && (cur_col>=0) && (cur_row<height) && (cur_col<width)){
                og_pos = width*cur_row + cur_col;
                filter_pos = f->dimension*(i+mid) +(j+mid);
                result += original[og_pos] * f->matrix[filter_pos];
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

/* Given a chunk_queue node, append it to our Queue ADT in FIFO order */
void chunk_enqueue(chunk_queue *image_chunk){
	if (q_back == NULL)	{ // Means we are adding the first chunk to the queue
		q_front = image_chunk;
		q_back = image_chunk;
	} else {				// Else we already have at least one element in the queue
		q_back->next = image_chunk;
		q_back = q_back->next;
	}
}

/* Dequeues an element from our Queue ADT in FIFO order. Returns chunk_queue node, else
	NULL if the Queue is empty */
chunk_queue *chunk_dequeue() {
	if (q_front == NULL){
		return NULL; // No more chunks to process
	}
	chunk_queue *temp = q_front;
	q_front = q_front->next;
	return temp;				// We free the memory for the chunk after processing
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
    int size = width*height;
    int row, col;
    int32_t i, result, smallest, largest;
    // largest and minimun for int32_t
    smallest = INT_MAX;
    largest = INT_MIN;
    for (i=0; i<size; i++){
        row = i/width;
        col = i%width;
        result = apply2d(f, original, target, width, height, row, col);
        if (result > largest){ 
            largest = result;
        }
        if (result < smallest){
            smallest = result;
        }
    }
    for (i=0; i<size; i++){
        normalize_pixel(target, i, smallest, largest);
    }
}

/****************** ROW/COLUMN SHARDING ************/

/* -- Horizontal Sharding -- */

/* Applies filter2d in a sharded_row manner */
void sharded_row_apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height, int32_t start_index, int32_t end_index, int32_t *smallest, int32_t *largest, int32_t id){

    int row, col;
    int32_t i, result;
    // largest and minimun for int32_t
    *smallest = INT_MAX;
    *largest = INT_MIN;
    for (i=start_index; i<end_index; i++){
        row = i/width;
        col = i%width;
        result = apply2d(f, original, target, width, height, row, col);
        if (result > *largest){ 
            *largest = result;
        }
        if (result < *smallest){
            *smallest = result;
        }
    }

}

/* Thread entry point for SHARDED_ROWS method */
void* sharded_rows(void *task){
	// get work attributes
    work *task_work = (work *) task;
    common_work *common = task_work->common;
    int id = task_work->id;

    // get common attributes
    const filter *f = common->f;
    const int32_t *original = common->original_image;
    int32_t *target = common->output_image;
    int32_t width = common->width;
    int32_t height = common->height;
    int32_t num_threads = common->max_threads;
	
	// Init local vars
	int32_t smallest = INT_MAX;
    int32_t largest = INT_MIN;
	int32_t rows_per_thread = 0, start_index = 0, end_index = 0;
		
	// Calculate the number of rows per thread
	if ((height % num_threads) != 0 && (id + 1 == num_threads)){	// If number of rows is not evenly divisible by number of threads, last thread is responsible for more rows
		rows_per_thread = height - ((num_threads - 1) * (height / num_threads));		// Last thread takes remaining number of rows in the image
	} else {
		rows_per_thread = height / num_threads;			// All n-1 threads take an even distribution of rows
	}
	
	// Calculate the starting index (of original image 1D array) for each thread
	if ((height % num_threads) != 0 && (id + 1 == num_threads)){	
		start_index = width * (height - rows_per_thread);		// Last thread may be responsible for more or less rows than all other n-1 threads
	} else {
		start_index = id * width * rows_per_thread;			// For all threads except the last, the starting index is simply the multiple of how many rows it is responsible for
	}
	
	// Calculate the end index (of original image 1D array) for each thread
	if (id + 1 == num_threads){	
		end_index = width * height;		// Last thread end index is simply the end of the image
	} else {
		end_index = (id * width) + (width * rows_per_thread);			// End index for all other n-1 threads
	}
	
	// Apply the filter
	sharded_row_apply_filter2d(f, original, target, width, height, start_index, end_index, &smallest, &largest, id);
	
	// wait for other threads
    pthread_barrier_wait(&(common->barrier));
    // add local maximum and minimum data to the global maximum and minimum
    modify_global_max_min(largest, smallest);
	
	return NULL;
}

/* -- Vertical Sharding -- */

/* SHARDED_COLUMNS_COLUMN_MAJOR */
/* Thread entry point for HARDED_COLUMNS_COLUMN_MAJOR method */
void* sharded_columns_column_major(void *task)    
{
    // get work attributes
    work *task_work = (work *) task;
    common_work *common = task_work->common;
    int id = task_work->id;

    // get common attributes
    const filter *f = common->f;
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

    // column major here, reverse order for row major
    int32_t i, j, result, smallest, largest;
    smallest = INT_MAX;
    largest = INT_MIN;
    for (j=start_col; j<end_col; j++){
        for (i=0; i<height; i++){
            result = apply2d(f, original, target, width, height, i, j);
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

/* SHARDED_COLUMNS_ROW_MAJOR */
/* Thread entry point for SHARDED_COLUMNS_ROW_MAJOR method */
void* sharded_columns_row_major(void *task)    
{    
    // get work attributes
    work *task_work = (work *) task;
    common_work *common = task_work->common;
    int id = task_work->id;

    // get common attributes
    const filter *f = common->f;
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
            result = apply2d(f, original, target, width, height, i, j);
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

/***************** WORK QUEUE *******************/

/* Responsible for splitting original image into chunks of size work_chunk x work_chunk */
void chunk_image(const int32_t *original, int32_t *target, int32_t width, int32_t height, int32_t work_chunk){
	int i = 0, x_chunk = 0, y_chunk = 0, row = 0, col = 0;
	int size = width * height; 		// size of original image
	
	while(i < size){
		// We are iterating through a 1D array that represents a 2D image.
		// We get the relative row and col for a specific index in the 1D representation
		row = i / width;
		col = i % width;
		
		// No precondition to be assumed other than work_chunk > 0.
		// In the case that work_chunk is greater than our image, we set it to the bounds
		// of our image.
		if(work_chunk > height) work_chunk = height;
		if(work_chunk > width) work_chunk = width;
		
		// Determine whether or not we can create a chunk of size chunk_size x chunk_size
		// Our image may not neccessarily be evenly divided by chunk size and in which
		// case we create a chunk as big as possible
		x_chunk = (((width-1) - col) >= work_chunk) ? work_chunk : width - col;
		y_chunk = (((height-1) - row) >= work_chunk) ? work_chunk : height - row;
		
		// Allocate memory for a node to be added to the queue
		chunk_queue *new_chunk = malloc(sizeof(chunk_queue));
		
		// Set the neccessary attributes
		new_chunk->start_index = i;
		new_chunk->x_chunk = x_chunk;
		new_chunk->y_chunk = y_chunk;
		new_chunk->next = NULL;
		
		chunk_enqueue(new_chunk);	// Add the chunk to our queue
		
		// Increment index
		i += x_chunk;	
		if(i % width == 0) i +=(width * (y_chunk - 1));
	}
}

/* Thread entry point for WORK_QUEUE method */
void* queue_work(void *task)
{
	// get work attributes
    work *task_work = (work *) task;
    common_work *common = task_work->common;
	
    // get common attributes
    const filter *f = common->f;
    const int32_t *original = common->original_image;
    int32_t *target = common->output_image;
    int32_t width = common->width;
    int32_t height = common->height;
	
	
	int32_t smallest, largest;
	
	chunk_queue *temp = NULL;
	int row = 0, col = 0, result = 0, start_index = 0, temp_index = 0;
	
	while(1){
		// Synchronously attempt to grab a tile to process
		pthread_mutex_lock(&queue_mutex);
		temp = chunk_dequeue();
		pthread_mutex_unlock(&queue_mutex);
		if(temp == NULL) break;		// Queue is empty, break out of loop and let the thread return
		
		// We proceed in a manner similar to sequential implementation for each image chunk
		smallest = INT_MAX;
		largest = INT_MIN;
		
		// Starting from the topleft most pixel in the image chunk,
		// iterate through the chunk and process the pixels
		start_index = temp->start_index;
		for(int j = 0; j < temp->y_chunk; j++){
			for(int k = 0; k < temp->x_chunk; k++){
				temp_index = start_index + (j * width) + k;
				row = temp_index / width;
				col = temp_index % width;
				result = apply2d(f, original, target, width, height, row, col);
				
				// Set local maximum/minimum processed pixel values
				if (result > largest){ 
					largest = result;
				}	
				if (result < smallest){
					smallest = result;
				}
			}
		}
		
		free(temp);			// Free memory allocated for image chunk
		modify_global_max_min(largest, smallest);		// Book keeping
	}
    return NULL;
}

/***************** MULTITHREADED ENTRY POINT ******/
void apply_filter2d_threaded(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads, parallel_method method, int32_t work_chunk)
{

    global_max = INT32_MIN;
    global_min = INT32_MAX;
	
	// Initialize shared thread primitives
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);
	pthread_mutex_init(&write_mutex, NULL);
	pthread_mutex_init(&queue_mutex, NULL);
	
	// Gather and set all common attributes between threads
    common_work *shared_work = malloc(sizeof(common_work));
    shared_work->f = f;
    shared_work->original_image = original;
    shared_work->output_image = target;
    shared_work->width = width;
    shared_work->height = height;
    shared_work->max_threads = num_threads;
    shared_work->barrier = barrier;
	shared_work->chunk_size = work_chunk;

    pthread_t workers[num_threads];
    // initialize work (struct) here
    work *task;
	
	// If we are proceeding by work_chunk method, we first need to split the image into chunks
	if (method == WORK_QUEUE) chunk_image(original, target, width, height, work_chunk);

    int i;
    for (i=0; i<num_threads; i++){

        // fulfill each task
        task = malloc(sizeof(work));
        task->common = shared_work;
        task->id = i;

        if (method == SHARDED_ROWS) {
			pthread_create(&workers[i], NULL, sharded_rows, (void *)task);   
        }
        else if (method == SHARDED_COLUMNS_COLUMN_MAJOR){
		    pthread_create(&workers[i], NULL, sharded_columns_column_major, (void *)task);    
        }
        else if (method == SHARDED_COLUMNS_ROW_MAJOR){
		    pthread_create(&workers[i], NULL, sharded_columns_row_major, (void *)task);    
        }
        else if (method == WORK_QUEUE) {
            pthread_create(&workers[i], NULL, queue_work, (void *)task);   
        }
        else{
            // method does not appear
            exit(1);
        }
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
        normalize_pixel(target, i, global_min, global_max);
    }

    // finish barrier here
    pthread_barrier_destroy(&barrier);
}

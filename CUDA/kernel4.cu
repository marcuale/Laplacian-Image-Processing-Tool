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

#include "kernels.h"

__global__ void kernel4(const int8_t *filter, int32_t dimension, 
	const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = index / width;
    // int column = index % width;

    int i, j, cur_row, cur_col, og_pos, filter_pos, row, column;
    int mid = dimension/2;
    int32_t result = 0;

    /** The number of threads **/
    int32_t num_threads = 512;

    int32_t cur = index;
    // only the selected threads do the calculation
    if (index < num_threads) {
        while (cur < width * height) {
            result = 0;
            row = cur / width;
            column = cur % width;
            for (i=-mid; i<=mid; i++){
                for (j=-mid; j<=mid; j++){
                    cur_row = row + j;
                    cur_col = column + i;
                    if ((cur_row>=0) && (cur_col>=0) && (cur_row<height) && (cur_col<width)){
                        og_pos = width*cur_row + cur_col;
                        filter_pos = dimension*(i+mid) +(j+mid);
                        result += input[og_pos] * filter[filter_pos];
                    }
                }
            }
            
            output[width*row + column] = result;

            cur += num_threads;
        }
    }

    __syncthreads();

}

__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
	int32_t smallest, int32_t biggest)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    /** The number of threads **/
    int32_t num_threads = 512;
    int32_t cur = index;
    if (index < num_threads) {
        while (cur < width * height) {
            if (cur < width * height) {
                image[cur] = ((image[cur] - smallest) * 255) / (biggest - smallest);
            }
            cur += num_threads;
        }
    }

    __syncthreads();
    
}

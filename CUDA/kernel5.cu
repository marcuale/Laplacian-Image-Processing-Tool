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

/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void kernel5(const int8_t *filter, int32_t dimension, 
   const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / width;
	int column = index % width;
	
	int i, j, cur_row, cur_col, og_pos, filter_pos;
	int mid = dimension/2;
	int32_t result = 0;
	if (index < width * height){		// If there are more threads than elements we don't want them to do any work
		// Same procedure as A2 apply2d() function
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
	}
	
	__syncthreads();
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
   int32_t smallest, int32_t biggest)
{
}

The project can be compiled running 'make' in the working directory. 


Then, the project can be run using the -i and -o flags as described in the README for the CPU version of this tool. Unlike that tool, there is much less custimzation, but varying CUDA kernals are supplied,
where some can be more useful depending on the image. Again, only PGM files are supported at this time.


The kernels designed as such:

Kernel 1 - One pixel per thread, column-major

Kernel 2 - One pixel per thread, row-major

Kernel 3 - Multiple pixels per thread, consecutive rows, row-major

Kernel 4 - Multiple pixels per thread, sequential access with a stride equal to the number of threads

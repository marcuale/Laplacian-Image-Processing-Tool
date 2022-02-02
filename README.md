# Laplacian-Image-Processing-Tool



This tool applies discrete Laplacian filters to detect edges in grey-scale images (currently only PGM formatted images are supported).

If you wish to use a CPU to process the images, the tool can be built by simply running 'make' in the working directory, and can be run with the following arguments from the CLI:

The input image: "-i input.pgm". This is the argument to specify which image you wish to process

The output image: "-o output.pgm". This is the resulting image after applying the filter to the input image (or the built-in image selected). This is an optional argument. If the output image is not specified, then no output image is produced. That is, the program still computes the resulting image in memory but does not write it to a file on disk and its contents will be lost once the program exits. This is a remnent of testing, but one that I have not yet changed (for now...)

The number of threads: "-n num_threads". You must vary the number of threads between 1 and the number of cores available on the machine

Timing enabled: "-t toggle_timing". Timing is enabled if toggle_timing is set to 1, and disabled if set to 0. (for performance nerds)

The filter: "-f filter_number" (More details below..)

The execution method: -m method_number". If the method is SEQUENTIAL, the num_threads parameter can either not be provided or simply ignored. (More details below..)

The work chunk: "-c chunksize". This is an optional argument, used if you use execution method (5), which is a Work-Pool parallel algorithm, as opposed to Data-Parallel (for methods (1)-(5)). For certain images and hardware, this might make more sense for you.

The following filters (to be used with the -f flag) provided are:
(1) 3x3 Laplacian
(2) 5x5 Laplacian
(3) 9x9 Laplacian of Gaussian

And the following methods of execution are provided: 
(1) SEQUENTIAL
(2) SHARDED_ROWS
(3) SHARDED_COLUMNS_COLUMN_MAJOR
(4) SHARDED_COLUMNS_ROW_MAJOR
(5) WORK_QUEUE


If you wish to use your GPU for processing the images, please see the CUDA folder.

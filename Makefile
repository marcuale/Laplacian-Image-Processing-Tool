CC=gcc
GCC_OPT = -O2 -Wall -Werror

%.o: %.c
	$(CC) -c -o $@ $< $(GCC_OPT)

main: very_big_sample.o very_tall_sample.o main.c pgm.c filters.c
	$(CC) $(GCC_OPT) main.c pgm.c filters.c very_big_sample.o very_tall_sample.o -o main.out -lpthread
	

pgm_creator:
	$(CC) $(GCC_OPT) pgm_creator.c pgm.c -o pgm_creator.out

run:
	./run-job-a2.sh
	
clean:
	rm *.o *.out

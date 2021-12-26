all: generate_dataset.c utility.c 
		gcc -o run generate_dataset.c utility.c -lm -I.



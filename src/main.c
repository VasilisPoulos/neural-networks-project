#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kmeans.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#define NUM_OF_CLUSTERS 10

typedef struct {
    double error;
    double x;
    double y;
} temp_kmeans;



void kmeans_generator();

int main()
{   
    kmeans_generator();
    return 0;
} 

void kmeans_generator(){
    float min_error[20] = {0};
    float min_x[20] = {0};
    float min_y[20] = {0};
    int pos = 0;
    for(int i = 0; i < 20; i++)
    {   
        sleep(1);
        temp_kmeans* temp;
        temp = kmeans(i);
        min_error[i] = temp->error;
        min_x[i] = temp->x;
        min_y[i] = temp->y;
        //printf("For repetition no.%d error is %f\n",i,min_error[i]);
    }

    for(int temp_pos = 1; temp_pos < 20; temp_pos++)
    {
        if(min_error[temp_pos] < min_error[pos] )
        {
            pos = temp_pos;
        }
    }
    printf("Minimum error is present at location %d for x = %f and y = %f and its value is %f.\n", pos,min_x[pos],min_y[pos], min_error[pos]);
    }


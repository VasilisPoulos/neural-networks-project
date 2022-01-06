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


typedef struct {
    double error;
    double x;
    double y;
} Temp_kmeans;

Temp_kmeans temp_kmeans[NUM_OF_CLUSTERS];



void kmeans_generator();

int main()
{   
    kmeans_generator();
    return 0;
} 

void kmeans_generator(){
    float min_error[20] = {0};
    float min_x[20][NUM_OF_CLUSTERS] = {0};
    float min_y[20][NUM_OF_CLUSTERS] = {0};
    int pos = 0;
    float total_min_error;
    for(int i = 0; i < 20; i++)
    {   
        sleep(1);
        Temp_kmeans* temp;
        temp = kmeans(i);
        for(int j = 0; j < NUM_OF_CLUSTERS; j++)
        {
        total_min_error += temp[j].error;
        min_x[i][j] = temp[j].x;
        min_y[i][j] = temp[j].y;
        printf("For repetition no.%d error is %f\n",j,temp[j].error);
        }
        min_error[i] = total_min_error;
        total_min_error = 0.0;
        //printf("For repetition no.%d error is %f\n",i,min_error[i]);
    }

    for(int temp_pos = 1; temp_pos < 20; temp_pos++)
    {
        if(min_error[temp_pos] < min_error[pos] )
        {
            pos = temp_pos;
        }
    }
    printf("Minimum error is present at location %d and its value is %f.\n", pos,min_error[pos]);
    for(int j = 0; j < NUM_OF_CLUSTERS; j++)
    {
        printf("Minimum error clusters at locations: x = %f and y = %f.\n", min_x[pos][j],min_y[pos][j]);
    }
}


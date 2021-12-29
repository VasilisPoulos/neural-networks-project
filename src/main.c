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

int main()
{   
    float min_error[20] = {0};
    int pos = 0;
    float min = 0.0;
    for(int i = 0; i < 20; i++)
    {   
        sleep(1);
        min = kmeans(i);
        min_error[i] = min;
        printf("For repetition no.%d error is %f\n",i,min_error[i]);
    }

    for(int temp_pos = 1; temp_pos < 20; temp_pos++)
    {
        if(min_error[0] > min_error[temp_pos] )
        {
            pos = temp_pos;
        }
    }
    printf("Minimum error is present at location %d and its value is %f.\n", pos, min_error[pos]);
    return 0;
} 


/* Example:  
 *    #include "clock_timer.h"
 *    . . .
 *    double start, finish, elapsed;
 *    . . .
 *    GET_TIME(start);
 *    . . .
 *    Code to be timed
 *    . . .
 *    GET_TIME(finish);
 *    elapsed = finish - start;
 *    printf("The code to be timed took %e seconds\n", elapsed);
*/


#include <windows.h>

double clock_gettime() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER end;
    double elapsedSeconds;
    QueryPerformanceFrequency(&frequency);
    
    QueryPerformanceCounter(&end);
    elapsedSeconds = (end.QuadPart) / (double)frequency.QuadPart;
    return(elapsedSeconds);
}

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
   now =  clock_gettime(); \
}
//now = time.tv_sec + time.tv_nsec/1000000000.0; //seconds
//now = (BILLION * time.tv_sec) + time.tv_nsec; //nanoseconds
//clock_gettime(CLOCK_MONOTONIC_RAW, &time); //para nao ter influencia do NTP

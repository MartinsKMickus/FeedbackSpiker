#include "code_measurements.h"
#include <windows.h>

LARGE_INTEGER frequency;
LARGE_INTEGER start;
LARGE_INTEGER end;
double time_unit = 1000;

void start_chronometer()
{
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
}

double stop_chronometer()
{
    QueryPerformanceCounter(&end);
    return time_unit * (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
}

void set_chronometer_unit_seconds()
{
    time_unit = 1;
}

void set_chronometer_unit_miliseconds()
{
    time_unit = 1000;
}

void set_chronometer_unit_microseconds()
{
    time_unit = 1000000;
}
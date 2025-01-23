cmake_minimum_required(VERSION 3.24)

# Project uses C and CUDA
project(FeedbackSpiker LANGUAGES C CUDA)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)

# Run tests
enable_testing() # Although tests can be launched with IDE this is required for command line

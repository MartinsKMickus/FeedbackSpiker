cmake_minimum_required(VERSION 3.10)

# Project uses C and CUDA
project(FeedbackSpiker LANGUAGES C CUDA)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)

# Run tests
enable_testing() # Although tests can be launched with IDE this is required for `make test`

# Add the PortAudio library manually
# add_library(portaudio STATIC IMPORTED)
# set_target_properties(portaudio PROPERTIES
#     IMPORTED_LOCATION "${portaudio_SOURCE_DIR}/lib/.libs/libportaudio.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${portaudio_SOURCE_DIR}/include"
# )
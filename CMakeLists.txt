cmake_minimum_required(VERSION 3.10)

# Project uses C and CUDA
project(FeedbackSpiker LANGUAGES C CUDA)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# TODO: Do not download if available in system
# Use FetchContent to download and build PortAudio
include(FetchContent)
FetchContent_Declare(
  portaudio
  GIT_REPOSITORY https://github.com/PortAudio/portaudio
  GIT_TAG        master
)
FetchContent_MakeAvailable(portaudio)

# Add the PortAudio library manually
# add_library(portaudio STATIC IMPORTED)
# set_target_properties(portaudio PROPERTIES
#     IMPORTED_LOCATION "${portaudio_SOURCE_DIR}/lib/.libs/libportaudio.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${portaudio_SOURCE_DIR}/include"
# )

# Add the executable and link with libraries
add_executable(FeedbackSpiker main.c)
add_library(audio_process SHARED audio_process.c)
target_link_libraries(audio_process PRIVATE portaudio)
target_link_libraries(FeedbackSpiker PRIVATE portaudio audio_process)



# TODO: Do not download if available in system
# Use FetchContent to download and build PortAudio
include(FetchContent)
FetchContent_Declare(
  portaudio
  GIT_REPOSITORY https://github.com/PortAudio/portaudio
  GIT_TAG        master
)
FetchContent_MakeAvailable(portaudio)

add_executable(FeedbackSpiker
    main.c
    utilities/text_formatter.c
    utilities/input_handler.c
    spiker_network.c
    spiker_network_gpu.cu
)
add_library(audio_process SHARED audio_process.c)
target_link_libraries(audio_process PRIVATE portaudio)
target_link_libraries(FeedbackSpiker PRIVATE portaudio audio_process)

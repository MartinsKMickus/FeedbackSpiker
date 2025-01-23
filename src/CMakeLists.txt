

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
set_property(TARGET FeedbackSpiker PROPERTY CUDA_ARCHITECTURES native) # Without this there is a warning CMAKE >3.18 (you can comment this out and see)
add_library(audio_process audio_process.c)
target_link_libraries(audio_process portaudio)
target_link_libraries(FeedbackSpiker audio_process)

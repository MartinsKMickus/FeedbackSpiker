# Feedback Spiker (IN PROGRESS)
Feedback spiker is environment learening tool that takes data from sensors (microphone, camera, etc.)
This data is then encoded by temporal coder and provided to SNN (<b>Simplified version for better performance</b>).
Neural network processes this data and returns response in form of a audio.
This tool works ir real time and simulates simple spiking neural network learning from environment around.
Neural network is able to receive instant feedback of what is its output therefore learning just as a baby would.

## Target
This tool explores principles of spiking neural networks by utilizing different computational techniques.
Tool provides information on how a spikikng neural network bahaves in such a scenario and is it able to learn without a context.

## Installation
Installation uses CMake and CUDA for neuron network processing. Code is primaly designed to work on Windows. It is recommended to use Visual Studio for compilation.

### Prerequisites
Most likely if you can install and correctly setup these prerequisites then you will also be able to continue with compilation of the code.
1. Windows SDK and compilers. Included in Visual Studio desktop development with C++ feature ([Visual-Studio](https://visualstudio.microsoft.com/))
2. CMake ([CMake-Downloads](https://cmake.org/download/))
3. CUDA toolkit ([CUDA-Toolkit-Homepage](https://developer.nvidia.com/cuda-downloads))
### Required libraries (should be installed by CMake)
1. PortAudio
### First time setup
These can be used within termina. Otherwise Visual Studio IDE can be used.
```sh
mkdir build
cd build
camke ..
```
To build (to be launched from build directory):
```sh
cmake --build .
```

## Troubleshooting

### JACK/ALSA/PkgConfig/PulseAudio libraries missing
This can happen (if it is WSL2 (I guess)). On Ubuntu run:
```sh
sudo apt-get update
sudo apt-get install libasound2-dev libjack-jackd2-dev libpulse-dev pkg-config
```
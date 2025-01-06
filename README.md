# Feedback Spiker (IN PROGRESS)
Feedback spiker is environment learening tool that takes data from sensors (microphone, camera, etc.)
This data is then encoded by temporal coder and provided to SNN (<b>Simplified version for better performance</b>).
Neural network processes this data and returns response in form of a audio.
This tool works ir real time and simulates simple spiking neural network learning from environment around.
Neural network is able to receive instant feedback of what is its output therefore learning just as a baby would.

## Target
This tool explores principles of spiking neural networks by utilizing different computational techniques.
Tool provides information on how a spikikng neural network bahaves in such a scenario and is it able to learn without a context.

## Some specifications
- Mono audio used

## Installation
Installation uses CMake to avoid cross-platform compilation issues.
### Platforms tested
- Ubuntu WSL2 on Windows.
- MacOS works without CUDA.
### Required libraries (should be installed by CMake)
1. PortAudio
### First time setup
```sh
sh build.sh
```
Then use `make` command within build folder after code changes. Binary files will be stored in `build/bin/`.

## Troubleshooting

### JACK/ALSA/PkgConfig/PulseAudio libraries missing
This can happen (if it is WSL2 (I guess)). On Ubuntu run:
```sh
sudo apt-get update
sudo apt-get install libasound2-dev libjack-jackd2-dev libpulse-dev pkg-config
```
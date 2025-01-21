
#include "input_handler.h"

#ifdef __linux__

#include <termios.h> // termios, TCSANOW, ECHO, ICANON
#include <unistd.h>  // STDIN_FILENO

unsigned char raw_mode_enabled = 0;
char pressed_key = 0;

void enableRawMode() {
    struct termios t;
    tcgetattr(STDIN_FILENO, &t);  // Get terminal attributes
    t.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &t); // Apply changes
}

void disableRawMode() {
    struct termios t;
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag |= (ICANON | ECHO); // Enable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
}

char get_char_if_pressed()
{
    if (!raw_mode_enabled)
    {
        enableRawMode();
        raw_mode_enabled = 1;
    }
    pressed_key = 0;
    read(STDIN_FILENO, &pressed_key, 1);
    disableRawMode();
    return pressed_key;
}

#endif
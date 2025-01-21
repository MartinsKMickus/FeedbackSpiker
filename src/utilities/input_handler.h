

#ifdef __linux__

extern unsigned char raw_mode_enabled;
extern char pressed_key;

void enableRawMode();
void disableRawMode();
char get_char_if_pressed();

#endif


extern int g_Width;
extern int g_Height;

/// <summary>
/// Fills pixels on screen. Clears before
/// </summary>
/// <param name="screen_buffer">Buffer to fill screen with (horizontal sync)</param>
void fill_white_pixels(const char * screen_buffer);

void init_screen();

void destroy_screen();
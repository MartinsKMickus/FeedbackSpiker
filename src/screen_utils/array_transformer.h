

#define SCREEN_PIXEL_TYPE char

extern SCREEN_PIXEL_TYPE* destination_screen;

int init_dest_screen(int W, int H);

void resize_2d_array(const SCREEN_PIXEL_TYPE* src, int srcW, int srcH,
    SCREEN_PIXEL_TYPE* dst, int dstW, int dstH);

void resize_2d_array_nearest(const SCREEN_PIXEL_TYPE* src, int srcW, int srcH,
    SCREEN_PIXEL_TYPE* dst, int dstW, int dstH);

void destruct_dest_screen();
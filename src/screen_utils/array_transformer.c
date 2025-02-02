#include <math.h>
#include <stdio.h>
#include "array_transformer.h"
#include <stdlib.h> // calloc

SCREEN_PIXEL_TYPE* destination_screen;

int init_dest_screen(int W, int H)
{
    destination_screen = calloc(W * H, sizeof(SCREEN_PIXEL_TYPE));
    return 0;
}

// Resize src (size srcW x srcH) into dst (size dstW x dstH).
// Both images are assumed to be stored in row-major order.
void resize_2d_array(const SCREEN_PIXEL_TYPE* src, int srcW, int srcH,
    SCREEN_PIXEL_TYPE* dst, int dstW, int dstH)
{
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    for (int j = 0; j < dstH; j++) {
        for (int i = 0; i < dstW; i++) {

            // Determine the source rectangle for this destination pixel.
            float srcX0 = i * scaleX;
            float srcY0 = j * scaleY;
            float srcX1 = (i + 1) * scaleX;
            float srcY1 = (j + 1) * scaleY;

            // Compute integer bounds (in the source image) that overlap the area.
            int xStart = (int)floor(srcX0);
            int yStart = (int)floor(srcY0);
            int xEnd = (int)ceil(srcX1);
            int yEnd = (int)ceil(srcY1);

            float pixelSum = 0.0f;
            float areaSum = 0.0f;

            // Loop over all source pixels that might contribute.
            for (int y = yStart; y < yEnd; y++) {
                // Calculate overlap in y-direction.
                float yOverlap = fmin(srcY1, y + 1.0f) - fmax(srcY0, (float)y);
                if (yOverlap <= 0) continue;

                for (int x = xStart; x < xEnd; x++) {
                    // Calculate overlap in x-direction.
                    float xOverlap = fmin(srcX1, x + 1.0f) - fmax(srcX0, (float)x);
                    if (xOverlap <= 0) continue;

                    float overlapArea = xOverlap * yOverlap;

                    // Safety check: make sure we don't read outside the source array.
                    if (x >= 0 && x < srcW && y >= 0 && y < srcH) {
                        pixelSum += src[y * srcW + x] * overlapArea;
                        areaSum += overlapArea;
                    }
                }
            }
            // Set destination pixel as the weighted average.
            dst[j * dstW + i] = (unsigned char)(pixelSum / areaSum);
        }
    }
}


void resize_2d_array_nearest(const SCREEN_PIXEL_TYPE* src, int srcW, int srcH,
    SCREEN_PIXEL_TYPE* dst, int dstW, int dstH)
{
    for (int j = 0; j < dstH; j++) {
        for (int i = 0; i < dstW; i++) {
            int srcX = i * srcW / dstW;
            int srcY = j * srcH / dstH;
            dst[j * dstW + i] = src[srcY * srcW + srcX];
        }
    }
}

void destruct_dest_screen()
{
    if (destination_screen != NULL)
    {
        free(destination_screen);
    }
}

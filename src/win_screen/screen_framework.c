#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include "screen_framework.h"

// Global variables for demonstration
static HBITMAP g_hDIBSection = NULL;
static void* g_pBits = NULL;
int     g_Width = 1500;
int     g_Height = 1500;
static HWND    g_hWnd = NULL;

int g_Running = TRUE;
int g_expecting_screen = 1;

// Forward declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void CreateDIB(HWND hWnd);
void FillPixels();

DWORD WINAPI WindowThreadProc(LPVOID lpParam)
{
    // 1. Get a module instance handle for the current process
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // 2. Register window class
    WNDCLASS wc = { 0 };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("ConsoleDIBDemoClass");

    if (!RegisterClass(&wc))
    {
        printf("RegisterClass failed!\n");
        return -1;
    }

    // 3. Create the window
    g_hWnd = CreateWindow(
        wc.lpszClassName,
        TEXT("Spiking Neuron Network"),
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        //WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        g_Width, g_Height, // window size
        NULL, NULL,
        hInstance, NULL);

    if (!g_hWnd)
    {
        printf("CreateWindow failed!\n");
        return -1;
    }

    // 4. Show the window
    ShowWindow(g_hWnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hWnd);

    // 5. Standard message loop
    MSG msg;
    while (g_Running) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                g_Running = FALSE;
                printf("Received WM_QUIT. Exiting message loop.\n");
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        g_expecting_screen = 1;
        Sleep(16);
    }

    // Normally we'd never get here unless we break from the loop,
    // but let's return anyway for completeness.
    return 0;
}

void init_screen()
{
    // Create a background thread to handle Win32 window + message loop
    HANDLE hThread = CreateThread(
        NULL,       // default security
        0,          // default stack size
        WindowThreadProc,
        NULL,       // thread parameter
        0,          // creation flags
        NULL
    );
}

void destroy_screen()
{
    g_Running = 0;
}

// Window Procedure
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
        // Create the DIB section on window creation
        CreateDIB(hWnd);
        return 0;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        if (g_hDIBSection && g_pBits)
        {
            BITMAP bm;
            GetObject(g_hDIBSection, sizeof(bm), &bm);

            HDC hMemDC = CreateCompatibleDC(hdc);
            HBITMAP hOldBmp = (HBITMAP)SelectObject(hMemDC, g_hDIBSection);

            // Blit from the memory DC to the window DC
            BitBlt(hdc,
                0, 0, g_Width, g_Height,
                hMemDC,
                0, 0, SRCCOPY);

            SelectObject(hMemDC, hOldBmp);
            DeleteDC(hMemDC);
        }

        EndPaint(hWnd, &ps);
    }
    return 0;

    case WM_KEYDOWN:
        // Press escape to quit
        if (wParam == VK_ESCAPE)
        {
            PostQuitMessage(0);
        }
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, message, wParam, lParam);
}

// Create the DIB section for direct pixel access
void CreateDIB(HWND hWnd)
{
    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = g_Width;
    bmi.bmiHeader.biHeight = -g_Height; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(hWnd);

    g_hDIBSection = CreateDIBSection(
        hdc,
        &bmi,
        DIB_RGB_COLORS,
        &g_pBits,
        NULL,
        0);

    ReleaseDC(hWnd, hdc);

    // Request a paint
    //InvalidateRect(hWnd, NULL, FALSE);
}

// Example function to fill the DIB
void fill_white_pixels(const char* screen_buffer)
{
    if (!g_pBits || !g_expecting_screen)
    {
        return;
    }
    uint32_t* pixels = (uint32_t*)g_pBits;
    char pixel;
    for (size_t i = 0; i < g_Width * g_Height; i++)
    {
        pixel = (char)screen_buffer[i];
        pixels[i] = (0xFF << 24) | (pixel << 16) | (pixel << 8) | pixel; // White or black
    }

    //size_t index = 0;
    //for (int y = 0; y < g_Height; ++y)
    //{
    //    for (int x = 0; x < g_Width; ++x)
    //    {
    //        index = x + y * g_Width;

    //        uint8_t r = (uint8_t)screen_buffer[index];
    //        uint8_t g = r;
    //        uint8_t b = r;
    //        // Typically BGRA in 32-bit DIB
    //        pixels[index] = (0xFF << 24) | (b << 16) | (g << 8) | r;
    //    }
    //}
    InvalidateRect(g_hWnd, NULL, FALSE);
    g_expecting_screen = 0;
}

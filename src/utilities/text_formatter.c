#include "text_formatter.h"
#include <stdio.h> // printf
#include <stdlib.h> // malloc, free
#include <string.h> // strlen

char *format_text(const char* message_type, const char *text, const char *color)
{
    // 4 is the length of "[]: "
    char *formatted_text = malloc(4 + strlen(message_type) + strlen(text) + strlen(color) + strlen(RESET) + 1);
    printf(formatted_text, "[%s%s%s]: %s", color, message_type, RESET, text); // Binary zero is added automatically
    return formatted_text;
}

void print_and_free(char *formatted_text)
{
    printf("%s", formatted_text);
    free(formatted_text);
}

void print_error(const char *text)
{
    char *formatted_text = format_text("ERROR", text, RED);
    print_and_free(formatted_text);
}

void print_warning(const char *text)
{
    char *formatted_text = format_text("WARNING", text, YELLOW);
    print_and_free(formatted_text);
}

void print_info(const char *text)
{
    char *formatted_text = format_text("INFO", text, CYAN);
    print_and_free(formatted_text);
}

void print_success(const char *text)
{
    char *formatted_text = format_text("SUCCESS", text, GREEN);
    print_and_free(formatted_text);
}

void print_error_unformatted(const char *text)
{
    printf("[ERROR]: %s", text);
}

void print_warning_unformatted(const char *text)
{
    printf("[WARNING]: %s", text);
}

void print_info_unformatted(const char *text)
{
    printf("[INFO]: %s", text);
}

void print_success_unformatted(const char *text)
{
    printf("[SUCCESS]: %s", text);
}

#pragma once

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"


/// @brief Formats the given text with the given color
/// @param text Text to be formatted
/// @param color Color of the text
/// @return Formatted text
char *format_text(const char* message_type, const char *text, const char *color);
void print_and_free(char *formatted_text);

// Formatted
void print_error(const char *text);
void print_warning(const char *text);
void print_info(const char *text);
void print_success(const char *text);

// Unformatted
void print_error_unformatted(const char *text);
void print_warning_unformatted(const char *text);
void print_info_unformatted(const char *text);
void print_success_unformatted(const char *text);
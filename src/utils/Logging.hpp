#pragma once
#include <cstdio>
#include <cstdarg>


namespace hrm
{
#ifdef NDEBUG
    #define LOG_DEBUG(fmt, ...) ((void)0)
#else
    #define LOG_DEBUG(fmt, ...) log_impl(LogLevel::Debug, fmt, ##__VA_ARGS__)
#endif

#define LOG_INFO(fmt, ...) log_impl(LogLevel::Info, fmt, ##__VA_ARGS__)

#define LOG_WARNING(fmt, ...) log_impl(LogLevel::Warning, fmt, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) log_impl(LogLevel::Error, fmt, ##__VA_ARGS__)


enum class LogLevel : int { Debug = 0, Info, Warning, Error };

inline void log_impl(LogLevel lv, const char* fmt, ...) {
    const char* lvl_str[] = { "DEBUG", "INFO", "WARN", "ERROR" };

    std::fprintf(stderr, "[%s]: ", lvl_str[static_cast<int>(lv)]);

    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
}

}


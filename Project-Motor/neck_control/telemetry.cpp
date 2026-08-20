#include "telemetry.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace neck_control {

void neckLog(const char* fmt, ...) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    char head[64];
    snprintf(head, sizeof(head), "[NeckCtrl %02d:%02d:%02d.%03ld] ",
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000);
    va_list ap;
    va_start(ap, fmt);
    printf("%s", head);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

} // namespace neck_control

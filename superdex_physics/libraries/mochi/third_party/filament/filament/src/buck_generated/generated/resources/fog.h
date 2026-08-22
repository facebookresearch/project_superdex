#ifndef FOG_H_
#define FOG_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t FOG_PACKAGE[];
}

#define FOG_FOG_OFFSET 0
#define FOG_FOG_SIZE 27369
#define FOG_FOG_DATA (FOG_PACKAGE + FOG_FOG_OFFSET)

#endif

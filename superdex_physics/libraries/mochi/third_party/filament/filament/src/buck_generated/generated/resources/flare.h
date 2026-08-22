#ifndef FLARE_H_
#define FLARE_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t FLARE_PACKAGE[];
}

#define FLARE_FLARE_OFFSET 0
#define FLARE_FLARE_SIZE 10448
#define FLARE_FLARE_DATA (FLARE_PACKAGE + FLARE_FLARE_OFFSET)

#endif

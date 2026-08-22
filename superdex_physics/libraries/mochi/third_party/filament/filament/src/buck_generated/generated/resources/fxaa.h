#ifndef FXAA_H_
#define FXAA_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t FXAA_PACKAGE[];
}

#define FXAA_FXAA_OFFSET 0
#define FXAA_FXAA_SIZE 14052
#define FXAA_FXAA_DATA (FXAA_PACKAGE + FXAA_FXAA_OFFSET)

#endif

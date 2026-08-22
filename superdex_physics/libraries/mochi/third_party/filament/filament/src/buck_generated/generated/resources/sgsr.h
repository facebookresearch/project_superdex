#ifndef SGSR_H_
#define SGSR_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t SGSR_PACKAGE[];
}

#define SGSR_SGSR1_OFFSET 0
#define SGSR_SGSR1_SIZE 29908
#define SGSR_SGSR1_DATA (SGSR_PACKAGE + SGSR_SGSR1_OFFSET)

#endif

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Converts a binary file to a C source file with the data as a const uint8_t array.

This is equivalent to Filament's resgen tool with the -c flag, but simpler
and doesn't require building the full resgen tool.

Usage:
    bin_to_c.py <input.bin> <output.c> <SYMBOL_NAME>

Example:
    bin_to_c.py bloom.bin bloom.c BLOOM
    
    Generates:
        #include <stdint.h>
        const uint8_t BLOOM_PACKAGE[] = { ... };
"""

import sys
import os


def bin_to_c(input_path: str, output_path: str, symbol_name: str) -> None:
    """Convert a binary file to a C source file."""
    with open(input_path, 'rb') as f:
        data = f.read()
    
    package_symbol = f"{symbol_name}_PACKAGE"
    
    with open(output_path, 'w') as f:
        f.write('#include <stdint.h>\n')
        f.write(f'const uint8_t {package_symbol}[] = {{\n')
        
        for i, byte in enumerate(data):
            if i % 16 == 0:
                f.write('    ')
            f.write(f'0x{byte:02x},')
            if i % 16 == 15:
                f.write('\n')
            else:
                f.write(' ')
        
        if len(data) % 16 != 0:
            f.write('\n')
        
        f.write('};\n')


def main() -> int:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.c> <SYMBOL_NAME>", file=sys.stderr)
        print(f"Example: {sys.argv[0]} bloom.bin bloom.c BLOOM", file=sys.stderr)
        return 1
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    symbol_name = sys.argv[3]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        return 1
    
    bin_to_c(input_path, output_path, symbol_name)
    return 0


if __name__ == '__main__':
    sys.exit(main())

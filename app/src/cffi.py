from cffi import FFI

ffi = FFI()

# Define a C function signature
ffi.cdef("""
    int printf(const char *format, ...);
""")

# Load the standard C library
C = ffi.dlopen(None)

# Call the C printf function
C.printf(b"Hello, %s!\n", ffi.new("char[]", b"world"))


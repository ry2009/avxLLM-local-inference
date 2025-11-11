set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER clang-17 CACHE FILEPATH "Clang 17 C compiler" FORCE)
set(CMAKE_CXX_COMPILER clang++-17 CACHE FILEPATH "Clang 17 C++ compiler" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_INIT "-O3 -ffp-contract=fast -fno-exceptions -fno-rtti")
set(CMAKE_C_FLAGS_INIT   "-O3 -ffp-contract=fast -fno-exceptions -fno-rtti")

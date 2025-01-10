cmake_minimum_required(VERSION 3.13)
project(cspir)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Get Clang's resource directory and make it available to the code
execute_process(
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -print-resource-dir
    OUTPUT_VARIABLE CLANG_RESOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-DCLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}")

message(STATUS "Using Clang resource directory: ${CLANG_RESOURCE_DIR}")

add_executable(cspir
    src/main.cpp
    src/parser.cpp
    src/spirv_generator.cpp
    src/types.h)

# Find Clang libraries
find_library(CLANG_TOOLING_LIB clang-cpp REQUIRED)

# Add compiler flags
target_compile_options(cspir PRIVATE
    -fexceptions
    -frtti
    -Wall
    -Wextra
)

# Link against the main Clang library
target_link_libraries(cspir PRIVATE
    ${CLANG_TOOLING_LIB}
    LLVM
)

# Add system libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(cspir PRIVATE
        pthread
        dl
        z
    )
endif()

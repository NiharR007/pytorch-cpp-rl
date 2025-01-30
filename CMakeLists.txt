cmake_minimum_required(VERSION 3.0)
project(CppRl
    LANGUAGES CXX
    VERSION 1.2.0
    DESCRIPTION "Reinforcement learning in C++ using PyTorch"
)
option(CPPRL_BUILD_TESTS "Whether or not to build the CppRl tests" OFF)
# Add these lines before your add_executable commands
add_definitions(-DDOCTEST_CONFIG_NO_DEBUGGER)
add_definitions(-DDOCTEST_CONFIG_DISABLE_EXCEPTIONS)
add_definitions(-DDOCTEST_CONFIG_NO_POSIX_SIGNALS)
add_definitions(-DDOCTEST_CONFIG_DISABLE_DEBUGGER_BREAK)
add_definitions(-DDOCTEST_CONFIG_NO_INLINE_ASM)


# Project-wide properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")


# Cppcheck
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CPPCHECK_ARGS
        --enable=warning
        --std=c++14
        --force
        --verbose
        --quiet
        --inline-suppr
        --error-exitcode=1
        --language=c++
        --config-exclude=${CMAKE_CURRENT_LIST_DIR}/src/third_party
        --config-exclude=${CMAKE_CURRENT_LIST_DIR}/lib
        -i${CMAKE_CURRENT_LIST_DIR}/example/lib
        --suppressions-list=${CMAKE_CURRENT_LIST_DIR}/CppCheckSuppressions.txt 
        -I ${CMAKE_CURRENT_LIST_DIR}/src 
        -I ${CMAKE_CURRENT_LIST_DIR}/include 
        -I ${CMAKE_CURRENT_LIST_DIR}/example 
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/example
    )

    add_custom_target(
        check
        COMMAND cppcheck ${CPPCHECK_ARGS}
        COMMENT "Running Cppcheck"
    )
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# Dependencies
## PyTorch
if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif()
endif (NOT TORCH_FOUND)

if(NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed -Wl,--no-as-needed")
endif()

if(APPLE)
    # Remove problematic linker flags
    string(REPLACE "--no-as-needed" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "--as-needed" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "--no-as-needed" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REPLACE "--as-needed" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    
    # Set ZMQ paths explicitly
    set(ZMQ_ROOT "/opt/homebrew/Cellar/zeromq/4.3.5_1")
    set(ZMQ_INCLUDE_DIR "${ZMQ_ROOT}/include")
    set(ZMQ_LIBRARY "${ZMQ_ROOT}/lib/libzmq.dylib")
    
    # Add ZMQ to include directories
    include_directories(${ZMQ_INCLUDE_DIR})
    
    # Update TORCH_LIBRARIES
    set(TORCH_LIBRARIES 
        "${TORCH_LIBRARIES}"
        "/Users/niharpatel/Desktop/libtorch/lib/libtorch_cpu.dylib"
        "/Users/niharpatel/Desktop/libtorch/lib/libc10.dylib"
        "/Users/niharpatel/Desktop/libtorch/lib/libkineto.a"
    )
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib;/Users/niharpatel/Desktop/libtorch/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# Define targets
add_library(cpprl STATIC "")
target_compile_definitions(cpprl PRIVATE DOCTEST_CONFIG_DISABLE)
option(CPPRL_BUILD_TESTS "Whether or not to build the CppRl tests" ON)
if (CPPRL_BUILD_TESTS)
    add_executable(cpprl_tests "")
endif(CPPRL_BUILD_TESTS)

# Enable all warnings
if(MSVC)
    target_compile_options(cpprl PRIVATE /W0)
else(MSVC)
    target_compile_options(cpprl PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

# Includes
set(CPPRL_INCLUDE_DIRS
    include
    src
    ${TORCH_INCLUDE_DIRS}
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
target_include_directories(cpprl PRIVATE ${CPPRL_INCLUDE_DIRS})
if (CPPRL_BUILD_TESTS)
    target_include_directories(cpprl_tests PRIVATE ${CPPRL_INCLUDE_DIRS})
endif(CPPRL_BUILD_TESTS)

# Linking
target_link_libraries(cpprl
    ${TORCH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${ZMQ_LIBRARY}
)

if (CPPRL_BUILD_TESTS)
    target_link_libraries(cpprl_tests
        cpprl
        ${TORCH_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        zmq
    )
endif(CPPRL_BUILD_TESTS)

# Example
option(CPPRL_BUILD_EXAMPLE "Whether or not to build the CppRl Gym example" ON)
if (CPPRL_BUILD_EXAMPLE)
    add_subdirectory(example)
endif(CPPRL_BUILD_EXAMPLE)

# Recurse into source tree
add_subdirectory(src)

# Set position independent code
set_target_properties(cpprl PROPERTIES POSITION_INDEPENDENT_CODE ON)


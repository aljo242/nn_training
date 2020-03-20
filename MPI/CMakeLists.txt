cmake_minimum_required (VERSION 3.9)
project (mpi_test_proj VERSION 1.0 LANGUAGES CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(mpi_test two_sided_mpi.cpp include/nn.h)

target_include_directories(mpi_test PRIVATE include)
target_link_libraries(mpi_test MPI::MPI_CXX)





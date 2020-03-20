cmake_minimum_required (VERSION 3.9)
project (mpi_test_proj VERSION 1.0 LANGUAGES CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(one_sided_test one_sided_test.cpp)

target_link_libraries(one_sided_test MPI::MPI_CXX)





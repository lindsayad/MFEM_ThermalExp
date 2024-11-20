cmake_minimum_required(VERSION 3.10)

project(mfemExample)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}; ${CMAKE_SOURCE_DIR}/cmake/modules/)
message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE, "Debug")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


set(NETCDF_PREFIX ${CMAKE_SOURCE_DIR}/../moose/libmesh/installed)
set(NETCDF_LIBS "-Wl,-rpath,${NETCDF_PREFIX}/lib -L${NETCDF_PREFIX}/lib -lnetcdf")

set(MFEM_DIR /home/bill/projects/mfem_install/mfem/build/)
set(HYPRE_DIR /home/bill/projects/mfem_install/hypre/)
set(METIS_DIR /home/bill/projects/mfem_install/metis/build/Linux-x86_64/)

set(NETCDF_INCLUDES ${NETCDF_PREFIX}/include)
set(HYPRE_INCLUDES ${HYPRE_DIR}/src/hypre/include)
set(METIS_INCLUDES ${METIS_DIR}/include)

include_directories(${MFEM_DIR} ${HYPRE_INCLUDES} ${METIS_INCLUDES} ${CMAKE_SOURCE_DIR}/include)


add_subdirectory(bins)


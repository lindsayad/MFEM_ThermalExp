cmake_minimum_required(VERSION 3.10)

project(mfemExample)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}; ${CMAKE_SOURCE_DIR}/cmake/modules/)
message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE, "Debug")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


set(NETCDF_PREFIX "${CMAKE_SOURCE_DIR}/../moose/libmesh/installed" CACHE PATH "Path to the NetCDF installation")
set(NETCDF_LIBS "-Wl,-rpath,${NETCDF_PREFIX}/lib -L${NETCDF_PREFIX}/lib -lnetcdf")

set(MFEM_DIR "/home/bill/projects/mfem_install/mfem/build/" CACHE PATH "Path to the MFEM installation")
set(HYPRE_DIR "/home/bill/projects/mfem_install/hypre/" CACHE PATH "Path to the Hypre installation")
set(METIS_DIR "/home/bill/projects/mfem_install/metis/build/Linux-x86_64/" CACHE PATH "Path to the Metis installation")
set(PETSC_DIR "/home/lindad/projects/moose-no-cuda/petsc")
set(PETSC_ARCH "arch-moose")

set(NETCDF_INCLUDES ${NETCDF_PREFIX}/include)
set(HYPRE_INCLUDES ${HYPRE_DIR}/src/hypre/include)
set(METIS_INCLUDES ${METIS_DIR}/include)
set(MFEM_INCLUDES ${MFEM_DIR}/include)
set(PETSC_INCLUDES "${PETSC_DIR}/include" "${PETSC_DIR}/${PETSC_ARCH}/include")

include_directories(${MFEM_INCLUDES} ${HYPRE_INCLUDES} ${METIS_INCLUDES} ${CMAKE_SOURCE_DIR}/include ${PETSC_INCLUDES})

add_subdirectory(bins)

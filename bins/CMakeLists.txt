file(GLOB_RECURSE example_src "*.cpp")
file(GLOB_RECURSE project_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE project_includes "${PROJECT_SOURCE_DIR}/include/*.hpp")

set (${PROJECT_NAME}_INCLUDE_DIRS "")
foreach (_incFile ${project_includes})
    get_filename_component(_dir ${_incFile} PATH)
    list (APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${_dir})
endforeach()
list (REMOVE_DUPLICATES ${PROJECT_NAME}_INCLUDE_DIRS)

add_executable(main ${project_src} main.cpp)
target_include_directories(main PUBLIC ${${PROJECT_NAME}_INCLUDE_DIRS})
target_link_directories(main PUBLIC ${MFEM_DIR})
target_link_libraries(main PUBLIC mfem MPI::MPI_CXX)

add_executable(temp ${project_src} temp.cpp)
target_include_directories(temp PUBLIC ${${PROJECT_NAME}_INCLUDE_DIRS})
target_link_directories(temp PUBLIC ${MFEM_DIR})
target_link_libraries(temp PUBLIC mfem MPI::MPI_CXX)



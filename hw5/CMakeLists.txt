cmake_minimum_required(VERSION 3.14)
project(hw5 LANGUAGES CXX)

find_package(UPCXX REQUIRED)

# netID 
set(NET_ID xxx CACHE STRING "Your netID")

# Packaging system
if (NOT ${NET_ID} STREQUAL xxx)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "CS5220${NET_ID}_hw5")
    install(FILES kmer_hash.cpp hash_map.hpp DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# Build the kmer_hash executable
add_executable(kmer_hash_19 kmer_hash.cpp)
target_link_libraries(kmer_hash_19 PRIVATE UPCXX::upcxx)
target_compile_definitions(kmer_hash_19 PRIVATE "KMER_LEN=19")

# Build the kmer_hash executable
add_executable(kmer_hash_51 kmer_hash.cpp)
target_link_libraries(kmer_hash_51 PRIVATE UPCXX::upcxx)
target_compile_definitions(kmer_hash_51 PRIVATE "KMER_LEN=51")

# Copy the job scripts
configure_file(job-kmer-hash job-kmer-hash COPYONLY)


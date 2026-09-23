configure_file(ThesisConfig.h.cmakein ThesisConfig.h)

set(THESIS_HEADERS
  Calc.h
  DataStructs.h
  Grid.h
  Init.h
  Melt.h
  Out.h
  Run.h
  Util.h
)
if(Thesis_ENABLE_MPI)
  list(APPEND THESIS_HEADERS MpiStructs.h)
endif()

file(GLOB THESIS_SOURCES GLOB *.cpp)

install(FILES ${THESIS_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(Thesis STATIC ${THESIS_SOURCES} ${THESIS_HEADERS})

target_include_directories(Thesis PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS Thesis
  EXPORT Thesis_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(Thesis::Thesis ALIAS Thesis)
target_compile_features(Thesis PUBLIC cxx_std_17)
target_link_libraries(Thesis PUBLIC OpenMP::OpenMP_CXX)

# Optional automatic differentiation: make Real an OTI number from cpp_oti_lib
# (see oti_scalar.h). Default OFF -> Real is plain double, build unchanged.
option(THESIS_ENABLE_OTI "Use cpp_oti_lib OTI numbers as the Real type (automatic differentiation)" OFF)
if(THESIS_ENABLE_OTI)
  # cpp_oti_lib is header-only and published as Sparrow
  # (https://github.com/ORNL-MDF/Sparrow). Without an explicit
  # -DTHESIS_OTI_INCLUDE_DIR, look for a checkout next to this repository.
  find_path(THESIS_OTI_INCLUDE_DIR otinum/otinum.hpp
    HINTS "${PROJECT_SOURCE_DIR}/../Sparrow/include"
          "${PROJECT_SOURCE_DIR}/../cpp_oti_lib/include"
          "${PROJECT_SOURCE_DIR}/../../cpp_oti_lib/include"
    DOC "Path to the cpp_oti_lib (Sparrow) include directory")
  if(NOT EXISTS "${THESIS_OTI_INCLUDE_DIR}/otinum/otinum.hpp")
    message(FATAL_ERROR
      "THESIS_ENABLE_OTI=ON but otinum/otinum.hpp was not found. Clone "
      "https://github.com/ORNL-MDF/Sparrow next to this repository, or pass "
      "-DTHESIS_OTI_INCLUDE_DIR=/path/to/Sparrow/include")
  endif()
  message(STATUS "Thesis Real type: OTI (cpp_oti_lib at ${THESIS_OTI_INCLUDE_DIR})")
  target_compile_definitions(Thesis PUBLIC THESIS_ENABLE_OTI)
  target_include_directories(Thesis PUBLIC $<BUILD_INTERFACE:${THESIS_OTI_INCLUDE_DIR}>)
else()
  message(STATUS "Thesis Real type: double")
endif()

if(Thesis_ENABLE_MPI)
  target_link_libraries(Thesis PUBLIC MPI::MPI_CXX)
endif()
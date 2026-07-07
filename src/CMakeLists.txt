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
set(THESIS_OTI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp_oti_lib/include"
    CACHE PATH "Path to the cpp_oti_lib include directory")
if(THESIS_ENABLE_OTI)
  message(STATUS "Thesis Real type: OTI (cpp_oti_lib at ${THESIS_OTI_INCLUDE_DIR})")
  target_compile_definitions(Thesis PUBLIC THESIS_ENABLE_OTI)
  target_include_directories(Thesis PUBLIC $<BUILD_INTERFACE:${THESIS_OTI_INCLUDE_DIR}>)
else()
  message(STATUS "Thesis Real type: double")
endif()

if(Thesis_ENABLE_MPI)
  target_link_libraries(Thesis PUBLIC MPI::MPI_CXX)
endif()
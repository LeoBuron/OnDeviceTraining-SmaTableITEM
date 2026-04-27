
# Host build wiring. Adds the core ODT libs plus the host-only extras
# (NPYLoader + CSVHelper) which depend on the filesystem and are not
# linked into MCU builds.
#
# SmaTable dataset backend selection: HOST always uses the NPY backend so
# Optuna trials can sweep folds at runtime via env vars. The baked backend
# is the MCU path (cmake/pico/targets/*.cmake).

target_sources(HOST PRIVATE
    ${CMAKE_SOURCE_DIR}/src/dataset/smatable_dataset_npy.c
)

target_link_libraries(HOST PRIVATE
    ${ODT_LIBS}
    ${ODT_LIBS_HOST_EXTRA}
    m
)

target_compile_definitions(HOST PRIVATE
    DEBUG_MODE_ERROR
)

target_include_directories(HOST PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/examples
)

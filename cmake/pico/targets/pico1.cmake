
target_sources(PICO1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/dataset/smatable_dataset_baked.c
)
target_link_libraries(PICO1 PRIVATE pico_stdlib ${ODT_LIBS})
target_include_directories(PICO1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/data
)

pico_enable_stdio_usb(PICO1 1)
pico_enable_stdio_uart(PICO1 0)
pico_add_extra_outputs(PICO1)
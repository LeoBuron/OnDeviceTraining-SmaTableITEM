
target_sources(PICO2_W PRIVATE
    ${CMAKE_SOURCE_DIR}/src/dataset/smatable_dataset_baked.c
)
target_link_libraries(PICO2_W PRIVATE pico_stdlib ${ODT_LIBS})
target_include_directories(PICO2_W PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/data
)

pico_enable_stdio_usb(PICO2_W 1)
pico_enable_stdio_uart(PICO2_W 0)
pico_add_extra_outputs(PICO2_W)
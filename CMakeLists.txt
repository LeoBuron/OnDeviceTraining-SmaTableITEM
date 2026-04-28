cmake_minimum_required(VERSION 3.16...4.1)

include(FetchContent)
include(cmake/build_targets.cmake)

# Select which example under src/examples/<name>.c is compiled into the
# executable. To add a new example, drop a .c file in src/examples/ and pass
# -DODT_EXAMPLE=<name>. The default keeps the long-standing PICO/STM32 flow
# building without any extra flag.
set(ODT_EXAMPLE "linear_regression" CACHE STRING
    "Name of the example (src/examples/<ODT_EXAMPLE>.c) to compile.")
set(ODT_EXAMPLE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/examples/${ODT_EXAMPLE}.c")
if(NOT ODT_PREPARE)
    if(NOT EXISTS "${ODT_EXAMPLE_SRC}")
        message(FATAL_ERROR
            "ODT_EXAMPLE='${ODT_EXAMPLE}' not found at ${ODT_EXAMPLE_SRC}. "
            "List available examples with: ls src/examples/*.c")
    endif()
endif()

if(${ODT_PICO_PLATFORM})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/pico_pre.cmake)
elseif(${ODT_HOST_PLATFORM})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/host/host_pre.cmake)
endif()

project(BA C ASM CXX)

if(${ODT_PREPARE})
    FetchContent_Declare(
            pico_sdk
            GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
            GIT_TAG 2.2.0
    )
    FetchContent_Declare(
            odt
            GIT_REPOSITORY https://github.com/es-ude/OnDeviceTraining
            # pinned: upstream main with GroupNorm + bias-less-conv optimizer fix
            GIT_TAG 3e768c7
    )


    FetchContent_Populate(
            pico_sdk
            QUIET
            GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
            GIT_TAG 2.2.0
            GIT_SUBMODULES_RECURSE TRUE

            SOURCE_DIR ${CMAKE_SOURCE_DIR}/pico-sdk/src
            BINARY_DIR ${FETCHCONTENT_BASE_DIR}/pico-sdk/build
            SUBBUILD_DIR ${FETCHCONTENT_BASE_DIR}/pico-sdk/subbuild

    )
    FetchContent_Populate(
            odt
            QUIET
            GIT_REPOSITORY https://github.com/es-ude/OnDeviceTraining
            # pinned: upstream main with GroupNorm + bias-less-conv optimizer fix
            GIT_TAG "3e768c7"
            GIT_SUBMODULES_RECURSE FALSE

            SOURCE_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/src
            BINARY_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/build
            SUBBUILD_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/subbuild
           )
else()

# Common ODT library targets linked into every platform. Extended with
# ODT_LIBS_HOST_EXTRA for host builds (NPYLoader + CSVHelper need libc
# filesystem calls not present on the MCU).
set(ODT_LIBS
    # tensor + quantization
    Tensor TensorApi TensorConversion Quantization QuantizationApi Distributions
    # common utilities
    Common Rounding RNG
    # layers
    Layer Linear LinearApi Relu ReluApi Softmax SoftmaxApi
    Conv1d Conv1dApi Conv1dTransposed MaxPool1d AvgPool1d AdaptiveAvgPool1d
    Pool1dApi AdaptivePool1dApi Dropout DropoutApi Flatten FlattenApi
    LayerNorm LayerNormApi QuantizationLayer
    LayerQuant StateDictApi
    GroupNorm GroupNormApi
    # loss functions
    LossFunction MSE CrossEntropy
    # optimizer
    Optimizer OptimizerApi Sgd SgdApi
    # inference + training loop
    InferenceApi TrainingLoopApi TrainingBatchDefault TrainingEpochDefault CalculateGradsSequential
    # storage + dataloader
    StorageApi DataLoader DataLoaderApi
    # arithmetic / misc (needed by GroupNorm + LayerConfigAccess pulled in at 3e768c7)
    ArithmeticType ExecuteOp Reduce LayerConfigAccess MemProfile
)

set(ODT_LIBS_HOST_EXTRA NPYLoader NPYLoaderApi CSVHelper)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/OnDeviceTraining/src)

if(${ODT_PICO_PLATFORM})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/pico_post.cmake)
elseif(${ODT_HOST_PLATFORM})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/host/host_post.cmake)
    target_compile_definitions(HOST PRIVATE
        MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/examples/data")
    # Plan 2 Audit: allow -DDEPTH_SWEEP_HIDDEN_LAYERS=<N> at configure time to
    # parameterize mlp_mnist_depth_sweep_host.c without modifying the source.
    if(DEFINED DEPTH_SWEEP_HIDDEN_LAYERS)
        target_compile_definitions(HOST PRIVATE
            DEPTH_SWEEP_HIDDEN_LAYERS=${DEPTH_SWEEP_HIDDEN_LAYERS})
    endif()
    # SmaTable L1 dataset-backend equivalence test. Opt-in (-DODT_L1_TEST=ON)
    # because it requires the prep step to have produced fold .h files.
    if(ODT_L1_TEST)
        add_subdirectory(tests)
    endif()
elseif(${ODT_TARGET} STREQUAL "STM32F756XG")
    add_subdirectory(cmake/stm32cubemx/stm32f756xg)
elseif(${ODT_TARGET} STREQUAL "STM32L476XG")
    add_subdirectory(cmake/stm32cubemx/stm32l476xg)
elseif(${ODT_TARGET} STREQUAL "STM32L4R5XI")
    add_subdirectory(cmake/stm32cubemx/stm32l4r5zi)
endif()

endif()

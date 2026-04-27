# L1 backend-equivalence test. Opt-in via -DODT_L1_TEST=ON.
#
# Compiles both dataset backends into the same executable using
# SMATABLE_FN_SUFFIX so their public symbols don't collide. Reads SMATABLE_*
# env vars (NPY backend) and compile-time -DSMATABLE_BAKED_*_HEADER (baked).
#
# Defaults below assume LOSO fold 0 of the synthetic test fixture
# data/smatable/folds/LOSO/fold_00_*.h. Override by setting the L1_DATA_DIR
# CMake variable and re-pointing the SMATABLE_BAKED_*_HEADER macros.

if(NOT ODT_HOST_PLATFORM)
    return()
endif()

set(L1_FOLD_SCHEME "LOSO" CACHE STRING "fold scheme used by L1 test")
set(L1_FOLD        "00"   CACHE STRING "fold index used by L1 test")
set(L1_BAKED_TRAIN "smatable/folds/${L1_FOLD_SCHEME}/fold_${L1_FOLD}_train.h" CACHE STRING "")
set(L1_BAKED_TEST  "smatable/folds/${L1_FOLD_SCHEME}/fold_${L1_FOLD}_test.h"  CACHE STRING "")

# SMATABLE_DATA_INCLUDE_DIR is the directory whose `smatable/` subdir
# contains the prep-generated fold headers. Default points at the repo's
# data/ tree; override on amplitude where prep writes to $HPC_HOME/data/.
set(SMATABLE_DATA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH
    "Directory containing the smatable/folds/<scheme>/fold_*.h headers.")

add_library(smatable_npy_l1 OBJECT
    ${CMAKE_SOURCE_DIR}/src/dataset/smatable_dataset_npy.c)
target_compile_definitions(smatable_npy_l1 PRIVATE
    SMATABLE_FN_SUFFIX=_npy
    DEBUG_MODE_ERROR)
target_include_directories(smatable_npy_l1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    $<TARGET_PROPERTY:HOST,INCLUDE_DIRECTORIES>)

add_library(smatable_baked_l1 OBJECT
    ${CMAKE_SOURCE_DIR}/src/dataset/smatable_dataset_baked.c)
target_compile_definitions(smatable_baked_l1 PRIVATE
    SMATABLE_FN_SUFFIX=_baked
    SMATABLE_BAKED_TRAIN_HEADER="${L1_BAKED_TRAIN}"
    SMATABLE_BAKED_TEST_HEADER="${L1_BAKED_TEST}"
    DEBUG_MODE_ERROR)
target_include_directories(smatable_baked_l1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${SMATABLE_DATA_INCLUDE_DIR}
    $<TARGET_PROPERTY:HOST,INCLUDE_DIRECTORIES>)

add_executable(dataset_l1_equiv
    ${CMAKE_SOURCE_DIR}/tests/dataset_backend_equiv.c)
target_link_libraries(dataset_l1_equiv PRIVATE
    smatable_npy_l1
    smatable_baked_l1
    ${ODT_LIBS}
    ${ODT_LIBS_HOST_EXTRA}
    m)
target_include_directories(dataset_l1_equiv PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    $<TARGET_PROPERTY:HOST,INCLUDE_DIRECTORIES>)

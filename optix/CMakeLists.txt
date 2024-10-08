add_library(ptx_files OBJECT kernel.cu)
target_include_directories(ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:ptx_files>"
  "-DVARIABLE_NAME=mainRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS ptx_files
  COMMENT "Convert ptx files to header file"
)

# # fixme: macro
add_library(shadow_ptx_files OBJECT shadow_ray_kernel.cu)
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(shadow_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(shadow_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET shadow_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET shadow_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(shadow_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  shadow_ray_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/shadow_ray_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:shadow_ptx_files>"
  "-DVARIABLE_NAME=shadowRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS shadow_ptx_files
  COMMENT "Convert ptx files to header file"
)

add_library(pathGen_ptx_files OBJECT path_gen_kernel.cu)
target_include_directories(pathGen_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(pathGen_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(pathGen_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(pathGen_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(pathGen_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET pathGen_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET pathGen_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(pathGen_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  pathGen_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/path_gen_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:pathGen_ptx_files>"
  "-DVARIABLE_NAME=pathGenRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS pathGen_ptx_files
  COMMENT "Convert ptx files to header file"
)

add_library(precom_ptx_files OBJECT precom_ray_kernel.cu)
target_include_directories(precom_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(precom_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(precom_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(precom_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(precom_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET precom_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET precom_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(precom_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  precom_ray_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/precom_ray_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:precom_ptx_files>"
  "-DVARIABLE_NAME=precomRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS precom_ptx_files
  COMMENT "Convert ptx files to header file"
)

add_library(vis_ptx_files OBJECT vis_ray_kernel.cu)
target_include_directories(vis_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(vis_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(vis_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(vis_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(vis_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET vis_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET vis_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(vis_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  vis_ray_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/vis_ray_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:vis_ptx_files>"
  "-DVARIABLE_NAME=visRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS vis_ptx_files
  COMMENT "Convert ptx files to header file"
)

# TODO
add_library(distributed_traversal_ptex_files OBJECT distributed_traversal_kernel.cu)
target_include_directories(distributed_traversal_ptex_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(distributed_traversal_ptex_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(distributed_traversal_ptex_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(distributed_traversal_ptex_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(distributed_traversal_ptex_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET distributed_traversal_ptex_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET distributed_traversal_ptex_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(distributed_traversal_ptex_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  distributed_traversal_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/distributed_traversal_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:distributed_traversal_ptex_files>"
  "-DVARIABLE_NAME=traversalSource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS distributed_traversal_ptex_files
  COMMENT "Convert ptx files to header file"
)

add_library(secondary_ptx_files OBJECT secondary_ray_kernel.cu)
target_include_directories(secondary_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(secondary_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(secondary_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(secondary_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(secondary_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET secondary_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET secondary_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(secondary_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  secondary_ray_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/render/secondary_ray_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:secondary_ptx_files>"
  "-DVARIABLE_NAME=secondaryRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS secondary_ptx_files
  COMMENT "Convert ptx files to header file"
)
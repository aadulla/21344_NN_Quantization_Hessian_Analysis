# header files
set (HEADER_FILES 
     ${CMAKE_CURRENT_SOURCE_DIR}/globals.h
     ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.h
     ${CMAKE_CURRENT_SOURCE_DIR}/hessian.h
     ${CMAKE_CURRENT_SOURCE_DIR}/net.h
     ${CMAKE_CURRENT_SOURCE_DIR}/q_matrix.h
     ${CMAKE_CURRENT_SOURCE_DIR}/quantizer.h 
     ${CMAKE_CURRENT_SOURCE_DIR}/train_test.h 
     ${CMAKE_CURRENT_SOURCE_DIR}/utils.h)

target_sources(${EXEC_NAME} PRIVATE ${HEADER_FILES})
target_sources(${LIB_NAME} PRIVATE ${HEADER_FILES})
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/*.h*")

find_package(Doxygen)

if (DOXYGEN_FOUND)

    doxygen_add_docs(
        docs
        "${HEADER_LIST}"
        "${CMAKE_CURRENT_SOURCE_DIR}/MarkdownFile.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include"
        )

    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile.docs
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
        )

else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation. Skipping docs generation.")
endif (DOXYGEN_FOUND)


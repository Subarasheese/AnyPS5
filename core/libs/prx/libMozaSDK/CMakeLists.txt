add_library(libMozaSDK SHARED EXCLUDE_FROM_ALL Export.cpp)
target_include_directories(libMozaSDK PRIVATE ${LIBS_INCLUDE_DIR})
configure_windows_unwind(libMozaSDK)

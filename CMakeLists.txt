cmake_minimum_required (VERSION 3.10)

project ("ncnnVxp")

set(APP_NAME "ncnnVxp")
set(DEVELOPER_NAME "XimikBoda")
set(RAM "3100") #Ram requirement (in kb)
set(IMSI "1234567890" CACHE STRING "Imsi of first SIM-card")
set(API "File SIM_card ProMng Camera") #Api for app, 
set(APPID "0") #App id, special values: -1 is personal (imsi), 0 is in developing
set(CERTID "1" CACHE STRING "") #Cert id use from you cert, 1 is personal (imsi)
set(CERT "none" CACHE STRING "Path to cert, used when APPID!=-1 and CERTID!=1")
option(BACKGROUND "App can work background" OFF)

set(MRE_SDK "$ENV{MRE_SDK}" CACHE STRING "MRE SDK dir")
set(TinyMRESDK "$ENV{TinyMRESDK}" CACHE STRING "TinyMRESDK dir")

if(BACKGROUND)
    add_definitions(-DSUPPORT_BG)
endif()

include(cmake/MreUtils.cmake)

add_subdirectory(resources)
add_subdirectory(mreapi)
add_subdirectory(run)

add_subdirectory(main)

set(NCNN_THREADS OFF CACHE BOOL "Disable threads" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "Disable Vulkan" FORCE)
set(NCNN_STDIO OFF CACHE BOOL "Disable STDIO" FORCE)
set(NCNN_STRING OFF CACHE BOOL "Disable string" FORCE)
set(NCNN_RUNTIME_CPU OFF CACHE BOOL "Disable CPU runtime detection" FORCE)
set(NCNN_PIXEL OFF CACHE BOOL "Disable pixel drawing" FORCE)
#set(Threads_FOUND TRUE CACHE BOOL "Mock Threads" FORCE)
#add_library(Threads::Threads INTERFACE IMPORTED)

add_subdirectory(ncnn)

#Full api list: Audio Camera Call TCP File HTTP Sensor SIM_card Record SMS_person SMS_SP BitStream Contact LBS MMS ProMng SMSMng Video XML Sec SysStorage Payment BT PUSH UDP SysFile
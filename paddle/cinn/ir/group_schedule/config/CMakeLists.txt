cinn_proto_library(filetileconfig_proto SRCS tileconfig_desc.proto)

core_gather_headers()

gather_srcs(cinnapi_src SRCS group_tile_config.cc)
gather_srcs(cinnapi_src SRCS database.cc)

foreach(header ${filetileconfig_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()

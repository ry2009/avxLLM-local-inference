set(PYINFENG_SOURCES
    bindings.cc)

pybind11_add_module(pyinfeng ${PYINFENG_SOURCES})
target_link_libraries(pyinfeng PRIVATE infeng_core)

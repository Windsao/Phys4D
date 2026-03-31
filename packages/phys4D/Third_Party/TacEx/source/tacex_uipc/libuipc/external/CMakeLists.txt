









uipc_init_submodule(muda)
add_subdirectory(muda)

add_library(muda::muda ALIAS muda)
set_target_properties(muda PROPERTIES FOLDER "external")

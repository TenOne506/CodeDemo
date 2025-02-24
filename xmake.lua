set_toolchains("clang")
set_toolset("cc", "clang")
set_toolset("cxx", "clang++")

target("printorder")
    set_kind("binary")
    add_files("printorder/printorder.cpp")

target("uniquelock")
    set_kind("binary")
    add_files("uniquelock/uniquelock.cpp")

target("classexample")
    set_kind("binary")
    add_files("classexample/classexample.cpp")


target("smartpoint")
    set_kind("binary")
    add_files("smartpoint/smartpoint.cpp")


---singleinstance/singleinstance.cpp

target("singleinstance")
    set_kind("binary")
    add_files("singleinstance/singleinstance.cpp")
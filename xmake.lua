set_toolchains("clang")
set_toolset("cc", "clang")
set_toolset("cxx", "clang++")
-- export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
-- export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
add_ldflags("-L/opt/homebrew/opt/llvm/lib")
add_cxxflags("-I/opt/homebrew/opt/llvm/include")
function define_target(name)
    target(name)
        set_kind("binary")
        set_languages("c++20")  -- 设置 C++ 版本为 C++20
        add_files(name .. "/*.cpp")  -- 添加源文件

end

-- function define_cudatarget(name)
--     target(name)
--         --add_rules("cuda")
--         set_kind("binary")
--         set_toolset("cuda", "nvcc")
--         -- 添加 CUDA 编译选项
--         add_cuflags("-rdc=true", {force = true})  -- 启用 Relocatable Device Code
--         add_cuflags("-arch=sm_89", {force = true})  -- 指定 CUDA 架构
--         --add_cuflags("-arch=sm_89", {force = true})
--         add_files(name .. "/*.cu")  -- 添加源文件
--         --add_cugencodes("native")
--         --set_policy("check.auto_ignore_flags", false)
--         --add_cugencodes("compute_86")
--         set_policy("check.auto_ignore_flags", false)
-- end



define_target("printorder")
define_target("uniquelock")
define_target("classexample")
define_target("smartpoint")
define_target("singleinstance")
define_target("consumer")
define_target("cas")
define_target("threaddetach")
define_target("packagetask")
define_target("sharedptr")
define_target("enableshared")
define_target("crtp")
define_target("sharedlock")
define_target("interview")
define_target("map")
define_target("threadlocal")
define_target("mystring")
define_target("deletedemo")
define_target("quicksort")
define_target("mergesort")
define_target("threadpool")
define_target("futurepromise")
define_target("handout")
define_target("jicheng")
define_target("printarray")
define_target("vector")
define_target("str")
define_target("treeorder")
define_target("nolockqueue")
--define_target("meituan")


-- define_cudatarget("cudagemm")
-- define_cudatarget("cudasoftmax")
-- define_cudatarget("cudatran")
-- define_cudatarget("cudarecucesum")
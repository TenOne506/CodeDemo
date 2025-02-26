set_toolchains("clang")
-- set_toolchains("cuda")
set_toolset("cc", "clang")
set_toolset("cxx", "clang++")
-- add_cuflags("-gencode arch=compute_89,code=sm_89")
-- rule("cuda")
--     set_extensions(".cu") -- 设置扩展名
--     on_buildcmd_file(function(target, batchcmds, sourcefile)
--         -- 应用nvcc进行编译
--         batchcmds:show_commands(true)
--         batchcmds:compile_command("nvcc -c " .. sourcefile .. " -o " .. path.join(target:objectdir(), path.basename(sourcefile) .. ".o"))
--     end)

function define_target(name)
    target(name)
        set_kind("binary")
        add_files(name .. "/*.cpp")  -- 添加源文件
end

function define_cudatarget(name)
    target(name)
        --add_rules("cuda")
        set_kind("binary")
        set_toolset("cuda", "nvcc")
        add_cuflags("-arch=sm_89", {force = true})
        add_files(name .. "/*.cu")  -- 添加源文件
        --add_cugencodes("native")
        --set_policy("check.auto_ignore_flags", false)
        --add_cugencodes("compute_86")
end



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


define_cudatarget("cuda")
# Project Carmen
This directory contains a C++ implementation of the Carmen storage system.

# Required Tools
To build the C++ implementation, you need the following tools:
 - A C++ compiler supporting C++20 (recommended clang 14+)
 - A matching C++ standard library (e.g., on Ubuntu `libc++-14-dev` and `libstdc++-12-dev`)
 - The Bazel build tool

## Installing Bazel
We recommend using `Bazelisk`, which can be installed as a Go tool using the following command:
```
go install github.com/bazelbuild/bazelisk@latest
```
Once installed, the `bazelisk` binary will be located in your Go tool folder. If `$GOPATH` is set, the binary should be found at `$GOPATH/bin/bazelisk`. If not, it will default to `~/go/bin/bazelisk`.

The `bazelisk` binary is a drop-in replacement for the `bazel` command that automatically fetches the required version of `bazel` according to the target project’s requirements. However, to make it accessible as the `bazel` command, you need to create a symbolic link.

To do so, pick a directory listed in your `$PATH` environment variable and create a symbolic link named `bazel` in this directory:
```
ln -s <path_to_bazelisk> bazel
```
For example, if `~/go/bin` is in your `$PATH`, and `bazelisk` was installed there, run:
```
ln -s ~/go/bin/bazelisk ~/go/bin/bazel
```
This will give you access to `bazel` on the command line.

## Installing Formatting Tools

### C++ File Formatting
Code files must be formatted according to the rules configured for `clang-format` in the repository. These formatting rules are checked and enforced during pull requests. To install `clang-format`, please use your platform-specific package manager.

### Bazel BUILD File Formatter
We recommend using `buildifier`, which can be installed as a Go tool:
```
go install github.com/bazelbuild/buildtools/buildifier@latest
```
Once installed, the `buildifier` binary will be found in your Go tool folder. If `$GOPATH` is set, the binary should be located at `$GOPATH/bin/buildifier`. If not, it will default to `~/go/bin/buildifier`.

# Build and Test
To build the entire project, use the following command:
```
bazel build //...
```

For this to work, you must have `bazel` installed on your system, as well as a C++ compiler toolchain supporting the C++20 standard. We recommend using clang.

To run all unit tests, use:
```
bazel test //...
```

You can build individual targets using commands like:
```
bazel run //common:type_test
```

# Profiling
To profile and visualize profiled data, we recommend using `pprof`. To install it as a Go tool, run:
```
go install github.com/google/pprof@latest
```

The binary will be installed in `$GOPATH/bin` (default is `$HOME/go/bin`). To make it accessible as the `pprof` command, create a symbolic link or alias:
```
alias pprof=<path_to_pprof>
```
To make the alias persistent, add it to your `.bashrc` or `.zshrc` file.

Link the profiler (`//third_party/gperftools:profiler`) into the target binary you want to profile.

Example:
```
cc_binary(
    name = "eviction_policy_benchmark",
    srcs = ["eviction_policy_benchmark.cc"],
    deps = [
        ":eviction_policy",
        "@com_github_google_benchmark//:benchmark_main",
        "//third_party/gperftools:profiler",
    ],
)
```
To start collecting profiling data, run the binary with the `CPUPROFILE` environment variable set to the path of the output file:
```
CPUPROFILE=/tmp/profile.dat bazel run -c opt //backend/store:store_benchmark -- --benchmark_filter=HashExponential.*File.*/16
```

To visualize the collected data (make sure `graphviz` is installed), run:
```
pprof --http=":8000" /tmp/profile.dat
```

# Setting Up Your IDE
The setup of your development environment depends on your choice of IDE.

## Visual Studio Code
To set up VS Code, install the following extensions:
 - [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)
 - [bazel-stack-vscode](https://marketplace.visualstudio.com/items?itemName=StackBuild.bazel-stack-vscode)
 - [bazel-stack-vscode-cc](https://marketplace.visualstudio.com/items?itemName=StackBuild.bazel-stack-vscode-cc)
 - [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
 - [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
 - [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

To open the project, use the *cpp* directory as the project root in VS Code.

Once everything is set up, open the command panel (Ctrl+Shift+P) and run the command:
```
Bazel/C++: Generate Compilation Database
```
This will generate a `compile_commands.json` file in the `cpp` directory listing local code dependencies pulled in by the Bazel build system. IntelliSense uses this file to locate source files of dependencies. This file is specific to your environment and should not be checked into the repository.

With this setup, VS Code should be ready to support editing code with proper code completion and navigation.

If you encounter any issues with this description, feel free to update it and submit a pull request.
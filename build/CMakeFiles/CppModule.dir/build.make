# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ducphong/CNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ducphong/CNN/build

# Include any dependencies generated for this target.
include CMakeFiles/CppModule.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CppModule.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CppModule.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CppModule.dir/flags.make

CMakeFiles/CppModule.dir/main.cpp.o: CMakeFiles/CppModule.dir/flags.make
CMakeFiles/CppModule.dir/main.cpp.o: ../main.cpp
CMakeFiles/CppModule.dir/main.cpp.o: CMakeFiles/CppModule.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ducphong/CNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CppModule.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CppModule.dir/main.cpp.o -MF CMakeFiles/CppModule.dir/main.cpp.o.d -o CMakeFiles/CppModule.dir/main.cpp.o -c /home/ducphong/CNN/main.cpp

CMakeFiles/CppModule.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CppModule.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ducphong/CNN/main.cpp > CMakeFiles/CppModule.dir/main.cpp.i

CMakeFiles/CppModule.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CppModule.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ducphong/CNN/main.cpp -o CMakeFiles/CppModule.dir/main.cpp.s

# Object files for target CppModule
CppModule_OBJECTS = \
"CMakeFiles/CppModule.dir/main.cpp.o"

# External object files for target CppModule
CppModule_EXTERNAL_OBJECTS =

CppModule.cpython-38-x86_64-linux-gnu.so: CMakeFiles/CppModule.dir/main.cpp.o
CppModule.cpython-38-x86_64-linux-gnu.so: CMakeFiles/CppModule.dir/build.make
CppModule.cpython-38-x86_64-linux-gnu.so: CMakeFiles/CppModule.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ducphong/CNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module CppModule.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CppModule.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CppModule.dir/build: CppModule.cpython-38-x86_64-linux-gnu.so
.PHONY : CMakeFiles/CppModule.dir/build

CMakeFiles/CppModule.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CppModule.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CppModule.dir/clean

CMakeFiles/CppModule.dir/depend:
	cd /home/ducphong/CNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ducphong/CNN /home/ducphong/CNN /home/ducphong/CNN/build /home/ducphong/CNN/build /home/ducphong/CNN/build/CMakeFiles/CppModule.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CppModule.dir/depend


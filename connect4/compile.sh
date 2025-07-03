#!/bin/bash

# Compile the Connect4 solver
g++ -O3 -std=c++11 -o connect4 main.cpp Solver.cpp Position.hpp

# Compile the position generator
g++ -O3 -std=c++11 -o generator generator.cpp Position.hpp OpeningBook.hpp

echo "Compilation complete. Created executables: connect4 and generator"
# 3d_rigid_body is a hybrid C++ and Python project for simulating and predicting 3D rigid body dynamics.
# It combines a C++ physics engine for accurate simulation with a PyTorch-based deep residual network
# for learning and predicting complex 3D object interactions and movements
#
# Copyright (c) 2024 Finbarrs Oketunji
# Written by Finbarrs Oketunji <f@finbarrs.eu>
#
# This file is part of 3d_rigid_body.
#
# 3d_rigid_body is an open-source software: you are free to redistribute
# and/or modify it under the terms specified in version 3 of the GNU
# General Public License, as published by the Free Software Foundation.
#
# 3d_rigid_body is is made available with the hope that it will be beneficial,
# but it comes with NO WARRANTY whatsoever. This includes, but is not limited
# to, any implied warranties of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. For more detailed information, please refer to the
# GNU General Public License.
#
# You should have received a copy of the GNU General Public License
# along with 3d_rigid_body.  If not, visit <http://www.gnu.org/licenses/>.

# Compiler and flags for C++
CXX = clang++
CXXFLAGS = -O3 -std=c++11 -Xpreprocessor -fopenmp -I/usr/local/include/eigen3
LDFLAGS = -lomp

# Directories
SRC_DIR = src
SCRIPT_DIR = scripts

# Target executable name
TARGET = 3d_rigid_body

# Source files
SRC = $(SRC_DIR)/main.cc $(SRC_DIR)/rigid_body.cc

# Header files
HEADERS = $(SRC_DIR)/rigid_body.h

# Python script
PYTHON_SCRIPT = $(SCRIPT_DIR)/residual_network.py

# Default target
all: install $(TARGET) run_python ## Build and run the entire project

# Install Eigen and Python packages
install: install_eigen install_python_packages ## Install Eigen and Python packages

install_eigen: ## Install Eigen on macOS
	brew install eigen

install_python_packages: ## Install Python Packages
	pip3 install -r requirements.txt

# Compile C++ program
$(TARGET): $(SRC) $(HEADERS) ## Compile the C++ program
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Execute C++ Program
run_cpp: $(TARGET) ## Execute the C++ Program
	./$(TARGET)

# Execute Python Script with default parameters
run_python: run_cpp ## Execute the Python script after the C++ Program
	python3 $(PYTHON_SCRIPT)

# Execute Python Script with custom parameters
run_python_custom: run_cpp ## Execute the Python script with custom parameters
	python3 $(PYTHON_SCRIPT) $(PYTHON_ARGS)

help: ## Display Help Message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all install install_eigen install_python_packages run_cpp run_python run_python_custom clean help
.DEFAULT_GOAL := help

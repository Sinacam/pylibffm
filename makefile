BUILD_DIR = pylibffm
SOURCE_DIR = src
LIBFFM_DIR = libffm

CXX ?= g++
CCFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -fopenmp
INCLUDE = $(shell python3-config --includes) -Ipybind11/include -Ilibffm

EXT_SUFFIX = $(shell python3-config --extension-suffix)
SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCE_DIR)/%.cpp,$(BUILD_DIR)/%$(EXT_SUFFIX),$(SOURCES))
LIBFFM_OBJECTS = $(LIBFFM_DIR)/ffm.o $(LIBFFM_DIR)/timer.o


all: $(OBJECTS)

$(BUILD_DIR)/%$(EXT_SUFFIX): $(SOURCE_DIR)/%.cpp $(LIBFFM_OBJECTS)
	$(CXX) $(CCFLAGS) $(INCLUDE) $< -o $@ $(LIBFFM_OBJECTS)

$(LIBFFM_OBJECTS):
	make -C libffm -f ../libffm-makefile

clean:
	make -C libffm -f ../libffm-makefile clean
	rm -f $(OBJECTS) $(LIBFFM_OBJECTS)
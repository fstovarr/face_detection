CC=g++
FLAGS=

IMG_FOLDER=img

EXECUTABLEDIR := bin
SOURCEDIR := src

SOURCES := $(SOURCEDIR)/main.cpp
EXECUTABLES := $(SOURCES:$(SOURCEDIR)/%.cpp=$(EXECUTABLEDIR)/%)

.PHONY: all prepare clean

all: clean prepare $(EXECUTABLES)

prepare:
	-@mkdir $(EXECUTABLEDIR)

$(EXECUTABLES): $(EXECUTABLEDIR)/%: $(SOURCEDIR)/%.cpp | $(EXECUTABLEDIR)
	$(CC) $< -o $@ $(FLAGS)

clean:
	-rm -R $(EXECUTABLEDIR)
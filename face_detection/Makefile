CC=g++
FLAGS=

IMG_FOLDER=img

EXECUTABLEDIR := bin
OUTPUTDIR := out
SOURCEDIR := src

SOURCES := $(SOURCEDIR)/main.cpp
EXECUTABLES := $(SOURCES:$(SOURCEDIR)/%.cpp=$(EXECUTABLEDIR)/%)
OUT_SUB_DIRS := $(SOURCES:$(SOURCEDIR)/%.cpp=$(OUTPUTDIR)/%)

.PHONY: all prepare clean fullclean

all: prepare $(EXECUTABLES)

prepare:
	-@mkdir $(EXECUTABLEDIR)
	-@mkdir $(OUTPUTDIR)
	-@mkdir $(OUT_SUB_DIRS)

$(EXECUTABLES): $(EXECUTABLEDIR)/%: $(SOURCEDIR)/%.cpp | $(EXECUTABLEDIR)
	$(CC) $< -o $@ $(FLAGS)

fullclean:
	-rm -R $(EXECUTABLEDIR)
	-rm -R $(OUTPUTDIR)
SRCDIR = $(shell pwd)
MODULE = server

CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++11 -g

SRCMAIN = $(SRCDIR)/main.cpp
SOURCES += $(shell find $(SRCDIR) -name '*.cpp' ! -wholename $(SRCMAIN))
BIN = banana-detector

OBJECTS  = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/$(MODULE)/%.o)
OBJMAIN  = $(SRCMAIN:$(SRCDIR)/%.cpp=$(OBJDIR)/$(MODULE)/%.o)

LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_imgcodecs

$(OBJDIR)/$(MODULE)/%.o:$(SRCDIR)/%.cpp
	mkdir -p $(@D)	# generate the directory
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $(@)

$(BINDIR)/$(BIN): $(OBJECTS) $(OBJMAIN)
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $(BINDIR)/$(BIN) $(OBJECTS) $(OBJMAIN) $(LIBS)
	@echo "Program built"

all: $(BINDIR)/$(BIN)

clean:
	rm $(BINDIR)/$(BIN)
	rm -rf $(OBJDIR)/$(MODULE)

test: $(BINDIR)/$(BIN)
	@$(BINDIR)/$(BIN)

mem: $(BINDIR)/$(BIN)
	valgrind --leak-check=yes $(BINDIR)/$(BIN)

debug: $(BINDIR)/$(BIN)
	gdb $(BINDIR)/$(BIN)

.PHONY: all clean test gtest mem debug

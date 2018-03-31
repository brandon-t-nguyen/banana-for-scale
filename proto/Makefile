export PROOT  = $(shell pwd)
export BINDIR = $(PROOT)/bin
export OBJDIR = $(PROOT)/obj
export LIBDIR = $(PROOT)/lib

export CC = gcc
export CXX = g++
export LD = ld
export AR = ar

export LIBS   = -L$(LIBDIR)
export INCLUDE = -I$(PROOT)/inc

BASE = src

banana-detector: $(LIB_DEPEND)
	+$(MAKE) -C $(BASE)/banana-detector

all: $(LIB_DEPEND)

clean:
	@#rm -rf cscope
	rm -rf $(BINDIR)
	rm -rf $(OBJDIR)
	rm -rf $(LIBDIR)
	+$(MAKE) clean -C external/libbtn

cscope:
	mkdir -p cscope
	tools/cscope_gen.sh -d $(PROOT)/inc -d $(PROOT)/src -o $(PROOT)/cscope -b q

run-detector: banana-detector
	$(BINDIR)/banana-detector --eyes_cascade=data/haarcascade_eye_tree_eyeglasses.xml --face_cascade=data/haarcascade_frontalface_alt.xml

dbg-detector: banana-detector
	gdb --args $(BINDIR)/banana-detector

.PHONY: all clean cscope server run-server client run-client

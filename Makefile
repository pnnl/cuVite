#### !!! USE cmake for building cuVite !!! ###
#### This Makefile is only used for ####
#### building the CPU-only version. ####

# change to CC for Cray systems
CXX = mpicxx

OPTFLAGS = -g -O3 -xHost -qopenmp -DDONT_CREATE_DIAG_FILES #-DDEBUG_PRINTF -DCHECK_COLORING_CONFLICTS
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 $(OPTFLAGS) #-DUSE_MPI_COLLECTIVES #-DUSE_32_BIT_GRAPH  #-DDEBUG_PRINTF

OBJFILES = main.o rebuild.o distgraph.o louvain.o coloring.o compare.o
BIN = bin

TARGET = $(BIN)/graphClustering

ALLTARGETS = $(TARGET)  

all: bindir $(ALLTARGETS)

bindir: $(BIN)
	
$(BIN): 
	mkdir -p $(BIN)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(GOBJFILES)
	$(CXX) $^ $(OPTFLAGS) -o $@ -lstdc++

.PHONY: bindir clean

clean:
	rm -rf *~ $(ALLOBJFILES) $(ALLTARGETS) $(BIN) dat.out.* check.out.*

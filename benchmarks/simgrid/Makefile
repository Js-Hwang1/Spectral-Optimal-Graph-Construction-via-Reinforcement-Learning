CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/opt/homebrew/Cellar/simgrid/4.1_1/include -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/Cellar/simgrid/4.1_1/lib -lsimgrid

all: dsgd_sim

dsgd_sim: dsgd_sim.cpp topologies.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f dsgd_sim

.PHONY: all clean

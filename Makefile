#
# Makefile for LFR and SNAP
#

LFR=./soft/lfr
SNAP = ./soft/snap

all:
	g++ -O3 -funroll-loops -o $(LFR)/benchmark $(LFR)/Sources/benchm.cpp
	$(MAKE) -C $(SNAP)/snap-core


clean:
	$(MAKE) clean -C $(SNAP)/snap-core
	rm -f $(LFR)/benchmark


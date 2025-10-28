# Makefile for Algorithm Testing

# Compiler and flags
CC = gcc
CFLAGS = -O3 -g -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable -I/opt/homebrew/Cellar/lapack/3.12.1/include
LDFLAGS = -L/opt/homebrew/Cellar/lapack/3.12.1/lib
LIBS = -lm -llapacke -llapack -lblas

# Source files
SOURCES_C2 = tests/Test_C2.c Phase1_src/Algorithm1.c Phase1_src/Special_Builder.c Phase1_src/C2_branch.c Phase1_src/C1_branch.c Phase1_src/B2_branch.c Phase1_src/B1_branch.c Phase1_src/A_branch.c Phase1_src/eigenvalue.c
SOURCES_C1 = tests/Test_C1.c Phase1_src/Algorithm1.c Phase1_src/Special_Builder.c Phase1_src/C1_branch.c Phase1_src/C2_branch.c Phase1_src/B2_branch.c Phase1_src/B1_branch.c Phase1_src/A_branch.c Phase1_src/eigenvalue.c
SOURCES_B2 = tests/Test_B2.c Phase1_src/Algorithm1.c Phase1_src/Special_Builder.c Phase1_src/B2_branch.c Phase1_src/C1_branch.c Phase1_src/B1_branch.c Phase1_src/C2_branch.c Phase1_src/A_branch.c Phase1_src/eigenvalue.c
SOURCES_B1 = tests/Test_B1.c Phase1_src/Algorithm1.c Phase1_src/Special_Builder.c Phase1_src/B1_branch.c Phase1_src/C1_branch.c Phase1_src/B2_branch.c Phase1_src/C2_branch.c Phase1_src/A_branch.c Phase1_src/eigenvalue.c

# Default target
all: test_c2 test_c1 test_b2 test_b1

test_c2: $(SOURCES_C2)
	$(CC) $(CFLAGS) $(LDFLAGS) -o bin/test_c2 $(SOURCES_C2) $(LIBS)

test_c1: $(SOURCES_C1)
	$(CC) $(CFLAGS) $(LDFLAGS) -o bin/test_c1 $(SOURCES_C1) $(LIBS)

test_b2: $(SOURCES_B2)
	$(CC) $(CFLAGS) $(LDFLAGS) -o bin/test_b2 $(SOURCES_B2) $(LIBS)

test_b1: $(SOURCES_B1)
	$(CC) $(CFLAGS) $(LDFLAGS) -o bin/test_b1 $(SOURCES_B1) $(LIBS)

clean:
	rm -f test_c2 test_c1 test_b2 test_b1 test_c2_real

.PHONY: all clean

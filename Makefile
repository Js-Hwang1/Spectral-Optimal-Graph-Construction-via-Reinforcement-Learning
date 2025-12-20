# =============================================================================
# Algebraic Connectivity Benchmark
# =============================================================================

CC      = clang
CFLAGS  = -O3 -Wall -Wextra -std=c11
SRCDIR  = src
DATADIR = data

# Platform-specific LAPACK linkage
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -framework Accelerate
else
    LDFLAGS = -llapack -lblas
endif
LDFLAGS += -lm

# Source files
SRCS = $(SRCDIR)/main.c \
       $(SRCDIR)/common.c \
       $(SRCDIR)/er.c \
       $(SRCDIR)/fv.c \
       $(SRCDIR)/sw.c \
       $(SRCDIR)/ours.c

OBJS = $(SRCS:.c=.o)
TARGET = benchmark

# =============================================================================
# Build targets
# =============================================================================

.PHONY: all clean run plot full help dirs clean-data

all: dirs $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c -o $@ $<

dirs:
	@mkdir -p $(DATADIR)

# =============================================================================
# Run targets
# =============================================================================

# Run with default N values (skips existing CSVs)
run: dirs $(TARGET)
	./$(TARGET)

# Run with custom N values (e.g., make run-custom N="64 128 256")
run-custom: dirs $(TARGET)
	./$(TARGET) $(N)

# Force re-run by deleting existing data first
run-force: clean-data run

# =============================================================================
# Plotting
# =============================================================================

# Plot using auto-detected N values from data/
plot:
	python3.11 plot.py

# Plot specific N values (e.g., make plot-custom N="32 64 96 128")
plot-custom:
	python3.11 plot.py $(N)

# Run everything: compile, benchmark, and plot
full: dirs $(TARGET)
	./$(TARGET)
	python3.11 plot.py

# =============================================================================
# Cleanup
# =============================================================================

# Remove compiled files only
clean:
	rm -f $(TARGET) $(OBJS)

# Remove data files only
clean-data:
	rm -f $(DATADIR)/*.csv

# Remove everything
clean-all: clean clean-data
	rm -f spectral_benchmark_grid.png

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Algebraic Connectivity Benchmark"
	@echo ""
	@echo "Project structure:"
	@echo "  src/       - Source files (main.c, common.c, er.c, fv.c, sw.c, ours.c)"
	@echo "  data/      - Output CSVs: ER_{N}.csv, FV_{N}.csv, OURS_{N}.csv, SW_r*_{N}.csv"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Compile the benchmark"
	@echo "  make run          - Run with defaults (32,64,96,128), skips existing CSVs"
	@echo "  make run-custom N='64 128'  - Run with custom N values"
	@echo "  make run-force    - Delete data and re-run all"
	@echo "  make plot         - Generate plot from available data/"
	@echo "  make plot-custom N='32 64'  - Plot specific N values"
	@echo "  make full         - Compile, run, and plot"
	@echo "  make clean        - Remove compiled files"
	@echo "  make clean-data   - Remove data/*.csv"
	@echo "  make clean-all    - Remove all generated files"
	@echo ""
	@echo "Output files per N:"
	@echo "  data/ER_{N}.csv       - Effective Resistance scores"
	@echo "  data/FV_{N}.csv       - Fiedler Vector scores"
	@echo "  data/OURS_{N}.csv     - Our algorithm scores"
	@echo "  data/SW_r25_{N}.csv   - Small World (rho=0.25)"
	@echo "  data/SW_r50_{N}.csv   - Small World (rho=0.50)"
	@echo "  data/SW_r75_{N}.csv   - Small World (rho=0.75)"

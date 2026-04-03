CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -fPIC -Wno-deprecated-declarations

SRCDIR = src
BINDIR = bin

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DACCELERATE_NEW_LAPACK
    # Use libomp if installed, otherwise fall back to single-threaded stubs
    ifneq ($(wildcard /opt/homebrew/opt/libomp/lib/libomp.dylib),)
        CFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
        LIB_LDFLAGS = -framework Accelerate -dynamiclib -L/opt/homebrew/opt/libomp/lib -lomp
        EXE_LDFLAGS = -framework Accelerate -L/opt/homebrew/opt/libomp/lib -lomp
    else
        CFLAGS += -I$(SRCDIR)/stubs
        LIB_LDFLAGS = -framework Accelerate -dynamiclib
        EXE_LDFLAGS = -framework Accelerate
    endif
    TARGET = $(BINDIR)/libcrl.dylib
    SHARED_FLAG = -dynamiclib
else
    LIB_LDFLAGS = -lopenblas -lm -lgomp
    EXE_LDFLAGS = -lopenblas -lm -lgomp
    CFLAGS += -fopenmp
    TARGET = $(BINDIR)/libcrl.so
    SHARED_FLAG = -shared
endif

.PHONY: all clean test train

all: $(TARGET) $(BINDIR)/crl_train $(BINDIR)/crl_eval $(BINDIR)/crl_sa $(BINDIR)/crl_v10 $(BINDIR)/crl_ga $(BINDIR)/crl_gnn $(BINDIR)/crl_mlp $(BINDIR)/crl_dqn

$(TARGET): $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) $(SHARED_FLAG) -o $@ $(SRCDIR)/crl.c $(LIB_LDFLAGS)

$(BINDIR)/crl_train: $(SRCDIR)/main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_eval: $(SRCDIR)/eval_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/eval_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_sa: $(SRCDIR)/sa_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/sa_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_v10: $(SRCDIR)/v10_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/v10_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_ga: $(SRCDIR)/ga_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/ga_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_saddle: $(SRCDIR)/saddle_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/saddle_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_enum: $(SRCDIR)/enumerate_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/enumerate_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

SADDLE_SRC = $(SRCDIR)/saddle/features.c $(SRCDIR)/saddle/climber.c \
             $(SRCDIR)/saddle/navigator.c $(SRCDIR)/saddle/episode.c \
             $(SRCDIR)/saddle/god_episode.c \
             $(SRCDIR)/saddle/train_climber.c $(SRCDIR)/saddle/train_navigator.c \
             $(SRCDIR)/saddle/weights.c $(SRCDIR)/saddle/main.c

$(BINDIR)/crl_saddle_rl: $(SADDLE_SRC) $(SRCDIR)/saddle/saddle.h $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SADDLE_SRC) $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_gnn: $(SRCDIR)/gnn_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/gnn_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_mlp: $(SRCDIR)/mlp_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/mlp_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_dqn: $(SRCDIR)/dqn_main.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/dqn_main.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR)/crl_test: $(SRCDIR)/test.c $(SRCDIR)/crl.c $(SRCDIR)/crl.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/test.c $(SRCDIR)/crl.c $(EXE_LDFLAGS)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR)

test: $(TARGET)
	python3 test_verify.py

train: $(BINDIR)/crl_train
	$(BINDIR)/crl_train $(ARGS)

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -fPIC -Wno-deprecated-declarations

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DACCELERATE_NEW_LAPACK -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LIB_LDFLAGS = -framework Accelerate -dynamiclib -L/opt/homebrew/opt/libomp/lib -lomp
    EXE_LDFLAGS = -framework Accelerate -L/opt/homebrew/opt/libomp/lib -lomp
    TARGET = libcrl.dylib
    SHARED_FLAG = -dynamiclib
else
    LIB_LDFLAGS = -lopenblas -lm -lgomp
    EXE_LDFLAGS = -lopenblas -lm -lgomp
    CFLAGS += -fopenmp
    TARGET = libcrl.so
    SHARED_FLAG = -shared
endif

.PHONY: all clean test train

all: $(TARGET) crl_train crl_eval crl_sa crl_v10

$(TARGET): crl.c crl.h
	$(CC) $(CFLAGS) $(SHARED_FLAG) -o $@ crl.c $(LIB_LDFLAGS)

crl_train: main.c crl.c crl.h
	$(CC) $(CFLAGS) -o $@ main.c crl.c $(EXE_LDFLAGS)

crl_eval: eval_main.c crl.c crl.h
	$(CC) $(CFLAGS) -o $@ eval_main.c crl.c $(EXE_LDFLAGS)

crl_sa: sa_main.c crl.c crl.h
	$(CC) $(CFLAGS) -o $@ sa_main.c crl.c $(EXE_LDFLAGS)

crl_v10: v10_main.c crl.c crl.h
	$(CC) $(CFLAGS) -o $@ v10_main.c crl.c $(EXE_LDFLAGS)

clean:
	rm -f $(TARGET) crl_train crl_eval crl_sa crl_v10 *.o

test: $(TARGET)
	python3 test_verify.py

train: crl_train
	./crl_train $(ARGS)

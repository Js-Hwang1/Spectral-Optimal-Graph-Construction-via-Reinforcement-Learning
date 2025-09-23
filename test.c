// Compile: mpicc -O2 -std=c11 test.c -o test
// Run:     mpirun -np 10 ./test [TAIL_N]


#define _XOPEN_SOURCE 700
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// ---------------- CONFIG ----------------
// Point this to your checkpoints directory (you said they're in ./runs)
static const char *MODELS_DIR  = "runs";
static const char *PYTHON_BIN  = "python3";
static const char *SCRIPT_PATH = "src/evaluate.py";
static const char *OUTPUT_DIR  = "output1";
static const char *GENEX  = "train_ema1";

// Hard-code any (n,m) you want tested:
static const int NM_PAIRS[][2] = {
{32,300}, {48,300}
};
static const int NM_PAIRS_COUNT = (int)(sizeof(NM_PAIRS)/sizeof(NM_PAIRS[0]));

enum { TAG_TASK=1, TAG_RESULT=2, TAG_TERMINATE=3 };

// -------------- UTIL --------------------
static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Exists but not a dir: %s\n", path);
            exit(1);
        }
        return;
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST) {
        perror("mkdir");
        fprintf(stderr, "Failed to create dir: %s\n", path);
        exit(1);
    }
}
static int has_suffix(const char *name, const char *suf) {
    size_t ln=strlen(name), ls=strlen(suf);
    return (ln>=ls && strcmp(name+(ln-ls), suf)==0);
}
static int has_prefix(const char *name, const char *pre) {
    size_t ln=strlen(name), ls=strlen(pre);
    return (ln>=ls && strncmp(name, pre, ls)==0);
}
static int is_regular_file(const char *path) {
    struct stat st;
    if (stat(path, &st)!=0) return 0;
    return S_ISREG(st.st_mode);
}
static int cmp_strptr(const void *a, const void *b) {
    const char *sa = *(const char * const *)a;
    const char *sb = *(const char * const *)b;
    return strcmp(sa, sb);
}
static const char* basename_const(const char *path) {
    const char *s = strrchr(path, '/');
    return s ? s+1 : path;
}
static int append_file(FILE *dst, const char *src_path) {
    FILE *src = fopen(src_path, "rb");
    if (!src) return -1;
    char buf[1<<15];
    size_t r;
    while ((r=fread(buf,1,sizeof(buf),src))>0) {
        if (fwrite(buf,1,r,dst)!=r) { fclose(src); return -2; }
    }
    fclose(src);
    return 0;
}
static void progress_line(int n, int m, int done, int total) {
    int width = 28;
    int filled = (int)((double)done / (double)total * width + 0.5);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    char bar[64];
    for (int i=0;i<width;i++) bar[i] = (i<filled)? '#':'.';
    bar[width] = '\0';
    double pct = total ? (100.0 * (double)done / (double)total) : 100.0;
    fprintf(stderr, "[MASTER] (%d,%d) %d/%d [%s] %5.1f%%\n", n, m, done, total, bar, pct);
}

// List .pt models, sorted ascending (lexicographic)
static char **list_pt_files(const char *dir, int *count_out) {
    DIR *dp = opendir(dir);
    if (!dp) { perror("opendir"); fprintf(stderr,"Cannot open %s\n", dir); exit(1); }
    char **list=NULL; int cap=0, n=0;
    struct dirent *de;
    while ((de=readdir(dp))!=NULL) {
        if (!strcmp(de->d_name,".")||!strcmp(de->d_name,"..")) continue;
        if (!has_suffix(de->d_name, ".pt")) continue;
        if (!has_prefix(de->d_name, GENEX)) continue;
        char path[PATH_MAX];
        snprintf(path,sizeof(path), "%s/%s", dir, de->d_name);
        if (!is_regular_file(path)) continue;
        if (n==cap) { cap = cap? cap*2:64; list = (char**)realloc(list, cap*sizeof(char*)); if(!list){perror("realloc"); exit(1);} }
        list[n] = strdup(path); if(!list[n]){perror("strdup"); exit(1);}
        n++;
    }
    closedir(dp);
    if (n>0) qsort(list, n, sizeof(char*), cmp_strptr);
    *count_out = n;
    return list;
}

// map (n,m) -> index in NM_PAIRS (or -1)
static int nm_index_of(int n, int m) {
    for (int i=0;i<NM_PAIRS_COUNT;i++) {
        if (NM_PAIRS[i][0]==n && NM_PAIRS[i][1]==m) return i;
    }
    return -1;
}
// map model path -> index in models[] (or -1)
static int model_index_of(char **models, int n_models, const char *path) {
    for (int i=0;i<n_models;i++) {
        if (strcmp(models[i], path)==0) return i;
    }
    return -1;
}

// -------------- TASK / RESULT -----------
typedef struct { int n, m; char model[PATH_MAX]; } Task;

typedef struct {
    int have;           // 1 if filled
    int exit_code;      // worker exit code
    int worker_rank;    // who ran it
    char tmpfile[PATH_MAX]; // temp file with captured output
} ResultCell;

// Build all tasks in a deterministic order: for each (n,m) in NM_PAIRS order,
// for each model in ascending order.
static Task *build_all_tasks(char **models, int n_models, int *count_out) {
    int total = NM_PAIRS_COUNT * n_models;
    Task *T = (Task*)malloc((size_t)total * sizeof(Task));
    if(!T){perror("malloc"); exit(1);}
    int t=0;
    for (int i=0;i<NM_PAIRS_COUNT;i++) {
        int n = NM_PAIRS[i][0], m = NM_PAIRS[i][1];
        for (int j=0;j<n_models;j++) {
            T[t].n = n; T[t].m = m;
            strncpy(T[t].model, models[j], PATH_MAX-1);
            T[t].model[PATH_MAX-1]='\0';
            t++;
        }
    }
    *count_out = total;
    return T;
}

// -------------- MASTER ------------------
static void master_send_task(int worker_rank, const Task *task) {
    char msg[PATH_MAX+64];
    snprintf(msg,sizeof(msg), "%d %d %s", task->n, task->m, task->model);
    MPI_Send(msg, (int)strlen(msg)+1, MPI_CHAR, worker_rank, TAG_TASK, MPI_COMM_WORLD);
}

// Receive one result, store in results[nm_idx][model_idx], update progress, return src rank.
static int master_recv_store_progress(ResultCell *results,
                                      int n_models, char **models,
                                      int *done_counts) {
    MPI_Status st;
    char buf[PATH_MAX*3 + 128];
    MPI_Recv(buf, sizeof(buf), MPI_CHAR, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &st);

    // Format: "n m <model_path> <tmpfile> <exit_code> <worker_rank>"
    int n,m, exit_code, worker_rank;
    char model[PATH_MAX], tmpfile[PATH_MAX];
    model[0]=tmpfile[0]='\0';

    if (sscanf(buf, "%d %d %4095s %4095s %d %d", &n, &m, model, tmpfile, &exit_code, &worker_rank) != 6) {
        fprintf(stderr, "[MASTER] Parse error for result: '%s'\n", buf);
        return st.MPI_SOURCE;
    }

    int ni = nm_index_of(n,m);
    int mj = model_index_of(models, n_models, model);
    if (ni<0 || mj<0) {
        fprintf(stderr, "[MASTER] Index lookup failed for n=%d m=%d model=%s\n", n, m, model);
        return st.MPI_SOURCE;
    }
    ResultCell *cell = &results[ni*n_models + mj];
    int first = !cell->have;
    cell->have = 1;
    cell->exit_code = exit_code;
    cell->worker_rank = worker_rank;
    strncpy(cell->tmpfile, tmpfile, PATH_MAX-1);
    cell->tmpfile[PATH_MAX-1]='\0';

    if (first) {
        done_counts[ni] += 1;
        progress_line(n, m, done_counts[ni], n_models);
    }
    return st.MPI_SOURCE;
}

// After all tasks are done, flush ordered logs per (n,m)
static void master_flush_logs(ResultCell *results, int n_models, char **models) {
    ensure_dir(OUTPUT_DIR);
    for (int i=0;i<NM_PAIRS_COUNT;i++) {
        int n = NM_PAIRS[i][0], m = NM_PAIRS[i][1];
        char out_path[PATH_MAX];
        snprintf(out_path,sizeof(out_path), "%s/output_%d_%d.log", OUTPUT_DIR, n, m);
        FILE *out = fopen(out_path, "wb"); // overwrite each run
        if (!out) { perror("fopen log"); fprintf(stderr,"Cannot open %s\n", out_path); continue; }

        for (int j=0;j<n_models;j++) {
            ResultCell *cell = &results[i*n_models + j];
            if (cell->have) {
                if (append_file(out, cell->tmpfile)!=0) {
                    fprintf(out, "[MASTER] WARNING: failed to append tmp: %s\n", cell->tmpfile);
                }
            } else {
                fprintf(out, "[MASTER] WARNING: missing result.\n");
            }
        }
        fclose(out);

        // Cleanup temp files
        for (int j=0;j<n_models;j++) {
            ResultCell *cell = &results[i*n_models + j];
            if (cell->have) unlink(cell->tmpfile);
        }
        fprintf(stderr, "[MASTER] Wrote %s\n", out_path);
    }
}

// -------------- WORKER ------------------
static int worker_recv_task(Task *task_out) {
    MPI_Status st;
    char msg[PATH_MAX+64];
    MPI_Recv(msg, sizeof(msg), MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    if (st.MPI_TAG == TAG_TERMINATE) return 0;
    if (st.MPI_TAG != TAG_TASK) return 0;
    int n,m; char model[PATH_MAX];
    if (sscanf(msg, "%d %d %4095s", &n, &m, model)!=3) return 0;
    task_out->n = n; task_out->m = m;
    strncpy(task_out->model, model, PATH_MAX-1);
    task_out->model[PATH_MAX-1]='\0';
    return 1;
}
static int worker_run_to_tmp(int my_rank, const Task *task, char *tmp_out, size_t cap) {
    ensure_dir(OUTPUT_DIR);
    snprintf(tmp_out, cap, "%s/tmp_rank%d_%d_%ld.log",
             OUTPUT_DIR, my_rank, getpid(), (long)time(NULL));

    char cmd[PATH_MAX*2];
    snprintf(cmd,sizeof(cmd),
             "%s %s  --load_model \"%s\" --n %d --m %d "
             "--topk 16",
             PYTHON_BIN, SCRIPT_PATH, task->model, task->n, task->m);

    FILE *fp = popen(cmd, "r");
    int exit_code = 0;
    FILE *out = fopen(tmp_out, "wb");
    if (!fp) {
        if (out) { fprintf(out, "[worker %d] ERROR: popen failed:\n%s\n", my_rank, cmd); fclose(out); }
        return 127;
    }
    char buf[1<<15];
    size_t r;
    while ((r=fread(buf,1,sizeof(buf),fp))>0) {
        if (out) fwrite(buf,1,r,out);
    }
    int status = pclose(fp);
    if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) exit_code = 128 + WTERMSIG(status);
    else exit_code = 1;
    if (out) fclose(out);
    return exit_code;
}
static void worker_send_result(int my_rank, const Task *task, const char *tmpfile, int exit_code) {
    char msg[PATH_MAX*3 + 128];
    snprintf(msg,sizeof(msg), "%d %d %s %s %d %d",
             task->n, task->m, task->model, tmpfile, exit_code, my_rank);
    MPI_Send(msg, (int)strlen(msg)+1, MPI_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
}

// -------------- MAIN --------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank==0) fprintf(stderr,"Need at least 2 ranks (1 master + >=1 worker)\n");
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        // MASTER
        ensure_dir(OUTPUT_DIR);

        int n_models = 0;
        char **models = list_pt_files(MODELS_DIR, &n_models);
        if (n_models == 0) {
            fprintf(stderr, "[MASTER] No .pt files in %s\n", MODELS_DIR);
            for (int r=1;r<size;r++) MPI_Send(NULL,0,MPI_CHAR,r,TAG_TERMINATE,MPI_COMM_WORLD);
            MPI_Finalize(); return 1;
        }

        // Optional: filter to last TAIL_N models if argv[1] provided
        if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
            char *endp = NULL;
            long tail = strtol(argv[1], &endp, 10);
            if (endp && *endp == '\0' && tail > 0) {
                if (tail < n_models) {
                    int start = n_models - (int)tail;
                    // Free models we will drop
                    for (int i = 0; i < start; i++) free(models[i]);
                    // Shift pointers down
                    memmove(models, models + start, (size_t)tail * sizeof(char*));
                    n_models = (int)tail;
                    fprintf(stderr, "[MASTER] Limiting to last %d models after sort.\n", n_models);
                } else {
                    fprintf(stderr, "[MASTER] TAIL_N (%ld) >= total models (%d); running all.\n", tail, n_models);
                }
            } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
                fprintf(stderr, "Usage: mpirun -np <procs> ./test [TAIL_N]\n");
                for (int r=1;r<size;r++) MPI_Send(NULL,0,MPI_CHAR,r,TAG_TERMINATE,MPI_COMM_WORLD);
                MPI_Finalize(); return 0;
            } else {
                fprintf(stderr, "[MASTER] Warning: ignoring non-numeric arg '%s'.\n", argv[1]);
            }
        }

        // Summarize plan
        fprintf(stderr, "[MASTER] Found %d models in %s\n", n_models, MODELS_DIR);
        fprintf(stderr, "[MASTER] Will run per (n,m):\n");
        for (int i=0;i<NM_PAIRS_COUNT;i++) {
            fprintf(stderr, "  - (%d,%d): %d models\n", NM_PAIRS[i][0], NM_PAIRS[i][1], n_models);
        }

        int total_tasks = 0;
        Task *tasks = build_all_tasks(models, n_models, &total_tasks);

        ResultCell *results = (ResultCell*)calloc((size_t)NM_PAIRS_COUNT * n_models, sizeof(ResultCell));
        if (!results){perror("calloc"); MPI_Abort(MPI_COMM_WORLD,1);}
        int *done_counts = (int*)calloc((size_t)NM_PAIRS_COUNT, sizeof(int));
        if (!done_counts){perror("calloc"); MPI_Abort(MPI_COMM_WORLD,1);}

        int num_workers = size - 1;
        int dispatched = 0, finished = 0;

        // Prime workers
        int init = (total_tasks < num_workers) ? total_tasks : num_workers;
        for (int k=0;k<init;k++) {
            master_send_task(1 + k, &tasks[dispatched++]);
        }

        // Dynamic loop
        while (finished < total_tasks) {
            int src = master_recv_store_progress(results, n_models, models, done_counts);
            finished++;
            if (dispatched < total_tasks) master_send_task(src, &tasks[dispatched++]);
        }

        // Terminate workers
        for (int r=1;r<size;r++) MPI_Send(NULL,0,MPI_CHAR,r,TAG_TERMINATE,MPI_COMM_WORLD);

        // Flush ordered logs
        master_flush_logs(results, n_models, models);

        for (int i=0;i<n_models;i++) free(models[i]);
        free(models);
        free(tasks);
        free(results);
        free(done_counts);

        fprintf(stderr, "[MASTER] Done. Logs in %s/\n", OUTPUT_DIR);

    } else {
        // WORKER
        for (;;) {
            Task T;
            if (!worker_recv_task(&T)) break;
            char tmpfile[PATH_MAX];
            int code = worker_run_to_tmp(rank, &T, tmpfile, sizeof(tmpfile));
            worker_send_result(rank, &T, tmpfile, code);
        }
    }

    MPI_Finalize();
    return 0;
}

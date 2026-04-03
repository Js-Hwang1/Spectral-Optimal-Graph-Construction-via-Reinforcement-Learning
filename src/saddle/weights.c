/*
 * weights.c -- Checkpoint save/load for climber and navigator agents.
 * Binary format: magic(4) + version(4) + episode(4) + reserved(20) + weights blob.
 */

#include "saddle.h"
#include <stdio.h>
#include <string.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t episode;
    uint32_t reserved[5];
} SdCkptHeader;

int save_climber(const char *path, const ClimberWeights *w, uint32_t episode) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    SdCkptHeader h = {0};
    h.magic = SD_MAGIC_CLIMBER;
    h.version = SD_CKPT_VERSION;
    h.episode = episode;
    fwrite(&h, sizeof(h), 1, f);
    fwrite(w, sizeof(ClimberWeights), 1, f);
    fclose(f);
    return 0;
}

int load_climber(const char *path, ClimberWeights *w, uint32_t *episode) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    SdCkptHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return -1; }
    if (h.magic != SD_MAGIC_CLIMBER) { fclose(f); return -2; }
    if (fread(w, sizeof(ClimberWeights), 1, f) != 1) { fclose(f); return -1; }
    if (episode) *episode = h.episode;
    fclose(f);
    return 0;
}

int save_navigator(const char *path, const NavigatorWeights *w, uint32_t episode) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    SdCkptHeader h = {0};
    h.magic = SD_MAGIC_NAVIGATOR;
    h.version = SD_CKPT_VERSION;
    h.episode = episode;
    fwrite(&h, sizeof(h), 1, f);
    fwrite(w, sizeof(NavigatorWeights), 1, f);
    fclose(f);
    return 0;
}

int load_navigator(const char *path, NavigatorWeights *w, uint32_t *episode) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    SdCkptHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return -1; }
    if (h.magic != SD_MAGIC_NAVIGATOR) { fclose(f); return -2; }
    if (fread(w, sizeof(NavigatorWeights), 1, f) != 1) { fclose(f); return -1; }
    if (episode) *episode = h.episode;
    fclose(f);
    return 0;
}

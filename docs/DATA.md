# Data: from the public source to the packed cache

The study reads one artefact: a packed 28x28 cache built from NIST Special
Database 19. This page is the whole path to it, and the numbers that prove you
built the same one we did.

**The data itself is not redistributed here.** SD19 is distributed by NIST under
its own terms; you download it from NIST. What this repository ships is the
conversion code, the exact command, and the checksums.

---

## 1. What you need

| File | Size | Why |
|---|---|---|
| `by_write.zip` | 568,113,446 B (~542 MB) | the images, organised by writer |
| `by_write_md5.log` | ~85 MB | MD5 of every by_write image |
| `by_class_md5.log` | ~140 MB | MD5 of every by_class image, which carries the **character label** |

`by_class.zip` itself (~4 GB) is **not needed**. Only its checksum log is: the
labels are recovered by joining the two logs on MD5, so every by_write image
gets its class without ever unpacking the second archive.

### Source

Landing page: <https://www.nist.gov/srd/nist-special-database-19>

Direct downloads (the S3 bucket NIST publishes the SRD from):

```bash
python tools/fetch_sd19.py --dest "$FOA_DATA_DIR/nist"
```

That fetches all three, **resumes** a partial download, **verifies** each
against the SHA-256 below, and prints the conversion command. It is idempotent:
a file already present and correct is reported and left alone, so it is safe to
re-run after an interruption. A file whose hash does not match is refused and
kept as `<name>.rejected` rather than deleted — a truncated transfer and a
different SD19 release are worth telling apart.

Only NIST's own host is built in. Pass your own mirror with `--mirror URL`
(repeatable, or `FOA_SD19_MIRRORS`); the checksum is what makes any mirror safe,
so an unverified one baked in would add risk and no convenience.

To verify what you already have without downloading anything:

```bash
python tools/fetch_sd19.py --dest "$FOA_DATA_DIR/nist" --check
```

By hand, if you prefer:

```bash
mkdir -p "$FOA_DATA_DIR/nist" && cd "$FOA_DATA_DIR/nist"
curl -fLO -C - https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
curl -fLO -C - https://s3.amazonaws.com/nist-srd/SD19/by_write_md5.log
curl -fLO -C - https://s3.amazonaws.com/nist-srd/SD19/by_class_md5.log
```

If the bucket is unavailable, the same archive is mirrored by several academic
hosts; verify whatever you obtain against the checksums below rather than
trusting the source. `foa`'s own downloader
(`federated_outlier_adaptation.data.download`) fetches the same three URLs and
is what the original run used.

### Verify what you downloaded

`study/UPSTREAM.sha256` carries **seven** entries: the three source files and
the four cache files, which do not exist yet at this point in the document. So
check the three by name rather than the file as a whole - a command that checks
all seven here fails on a correct download, and piping it through `head` hides
the four failures rather than answering them.

```bash
cd "$FOA_DATA_DIR/nist"
grep -E ' (by_write\.zip|by_write_md5\.log|by_class_md5\.log)$' \
    /path/to/repo/study/UPSTREAM.sha256 | sha256sum -c -
```

Three `OK` lines and exit 0. After section 2 has built the cache, the whole file
checks in one go, from the cache directory:

```bash
cd "$FOA_NIST28_DIR" && sha256sum -c /path/to/repo/study/UPSTREAM.sha256
```

or by hand:

```
39958e28827eb0d7d54f7e4c31c6cc36689b38aa218a4fc1e810c5413e7a35b8  by_write.zip
11e20fff1a3b934270b0b4dfe3e7928e933c2973ae0ec17938d0456be95a8c03  by_write_md5.log
b2a76dfb555a1fc3764672bb8514455727d28fbdd18df4ce5abbd85a879f430c  by_class_md5.log
```

These are the bytes the published results were produced from. A mismatch means
you have a different release of SD19, and nothing below will reproduce.

---

## 2. Build the packed cache

**One command. No GPU. The archive is never extracted.**

```bash
foa prepare-data \
    --dataset nist \
    --zip   "$FOA_DATA_DIR/nist/by_write.zip" \
    --out   "$FOA_NIST28_DIR" \
    --resolution 28 \
    --classes all
```

On a scheduler: `sbatch slurm/prepare_nist28.sbatch` (4 CPUs, 32 GB, no GPU;
it took **10.5 minutes** on our hardware, and it logs a projected total after
its first 2,000 images so you know the real cost within the first minute).
Pass `--output="$FOA_PROJECT_DIR/logs/%x_%j.log"` as well, or Slurm writes the
job log into whatever directory you submitted from.

> `--resolution 28` is not optional and is not the default. The conversion
> defaults to 128 px, which is the setting of an earlier study. A 128 px cache
> will train, produce numbers, and reproduce nothing on this page.

`--classes all` builds the full 62-class cache. The digit study then selects the
10-class subset at run time with `--classes digits`, so **one cache serves
both** and you never build it twice.

### What the conversion does

Per image: PNG decode -> Gaussian blur -> bounding-box crop -> bicubic resize to
28x28, the EMNIST preparation. The result is four files, whatever the sample
count:

| File | Contents |
|---|---|
| `nist28_images.npy` | uint8 `[N, 28, 28]` |
| `nist28_labels.npy` | uint8 `[N]`, 0-61 |
| `nist28_writers.npy` | writer id per row |
| `nist28_index.json` | class map, counts, source archive, conversion record |

Four files rather than 814,255 loose PNGs is deliberate: on a shared file system
with an inode quota, extracting SD19 is what fails, not what is slow.

---

## 3. Verify the cache — the numbers that must match

```bash
python - <<'PY'
import json, numpy as np, collections, os
D = os.environ["FOA_NIST28_DIR"]
lab = np.load(f"{D}/nist28_labels.npy")
wr  = np.load(f"{D}/nist28_writers.npy", allow_pickle=True)
c   = collections.Counter(lab.tolist())
dig = sum(v for k, v in c.items() if k < 10)
w   = np.array(wr)
print("rows            ", len(lab))
print("writers         ", len(set(wr.tolist())))
print("classes         ", len(c))
print("digit rows      ", dig)
print("digit writers   ", len(set(w[lab < 10].tolist())))
PY
```

Expected, exactly:

```
rows             814255
writers          3597
classes          62
digit rows       402953
digit writers    3580
```

### Byte-level equivalence

If all five counts match, you have the same index. To prove the pixels are
identical too:

```bash
cd "$FOA_NIST28_DIR" && sha256sum nist28_*.npy nist28_index.json
```

```
52ac9ac79a6149a69505eb7fad02309eedd7bf61437d1b30342f686d55d53b0f  nist28_images.npy
66ed05d90aad2d31e01c94aadfc458fcf4ad3f4407bc3845ace8792cbf868aaf  nist28_labels.npy
55a72acff7304b0c3c6f694e662254995092352c9fecfad15d8db2c45f04e943  nist28_writers.npy
00a12c6a51d89bb7e606069afb1c37db4f1e7935975f978857978d3d94b6edb3  nist28_index.json
```

`nist28_index.json` records the source archive path and the conversion
parameters, so its hash will differ if you built from a different location even
when the pixels match. **`nist28_images.npy` is the one that matters**: if that
hash matches, your cache is ours.

---

## 4. If you cannot download SD19

Much of this repository can still be checked. The test suite runs on synthetic
fixtures and needs no SD19 and no GPU - it prints its own count - and every task
file, selection record and fold book in `study/` can be verified against the
shipped checksums. See `docs/VERIFY.md` for what is checkable without the data.

There is **no** documented command that builds a small subset cache from
`by_write.zip`: `foa prepare-data` converts the whole archive or nothing. The
minutes-long smoke path in the quickstart therefore uses synthetic fixtures,
not a trimmed SD19.

## 5. Licence

NIST Special Database 19 is redistributed by NIST under the terms on its
landing page. This repository contains **no SD19 data** — only code, task
definitions, checksums, and small derived artefacts (writer lists, fold
assignments, accuracy tables) that do not contain images.

#!/usr/bin/env python3
"""Convert export_event.C's compact ROOT `tracks` TTree into one tracks-only NumPy archive

All 3D coordinates are stored in ALICE centimetres in ROOT and transformed
into Blender metres (X,Y,Z)=(z,x,y)/100 in the NPZ files
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import ROOT


def convert(input_file, output_prefix):
    root_file = ROOT.TFile.Open(str(input_file))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Cannot open {input_file}")
    tree = root_file.Get("tracks")
    if not tree:
        raise ValueError("Missing tracks tree")
    candidate = root_file.Get("candidate")
    metadata = {}
    if candidate and candidate.GetEntries() > 0:
        candidate.GetEntry(0)
        for key in ("candidateId", "mode", "run", "dataframe", "bc", "collision",
                    "contributors", "nBarrel", "nForward", "timestampMs"):
            metadata[key] = int(getattr(candidate, key))
        metadata["dataframeName"] = str(candidate.dataframeName)


    track_id, run, dataframe, aod_track_row, collision_id = [], [], [], [], []
    global_bc, bc_source, detector_mask, charge, pt_gev, eta = [], [], [], [], [], []
    offsets, paths = [0], []

    for t in tree:
        tid = int(t.trackId)
        track_id.append(tid)
        run.append(int(t.run))
        dataframe.append(int(t.df))
        aod_track_row.append(int(t.aodTrackRow))
        collision_id.append(int(t.collision))
        global_bc.append(int(t.bc))
        bc_source.append(int(t.bcSource))
        detector_mask.append(int(t.detectorMask))
        charge.append(int(t.charge))
        pt_gev.append(float(t.pt))
        eta.append(float(t.eta))

        # PyROOT may expose std::vector<float> storage as a NumPy VIEW
        # GetEntry() can reuse that buffer for the next track. Copy first
        path = np.asarray(t.xyz, dtype=np.float32).reshape(-1, 3).copy()
        paths.append(path)
        offsets.append(offsets[-1] + len(path))

    def blender(xyz):
        a = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        return a[:, [2, 0, 1]] * .01

    ids = np.asarray(track_id, dtype=np.int32)
    bcs = np.asarray(global_bc, dtype=np.uint64)
    collisions = np.asarray(collision_id, dtype=np.int32)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    extra = {"candidate_" + key: np.asarray(value, dtype=np.int64 if isinstance(value, int) else str)
             for key, value in metadata.items()}
    np.savez_compressed(
        str(prefix) + "_tracks.npz",
        track_id=ids,
        run=np.asarray(run, dtype=np.int32),
        dataframe=np.asarray(dataframe, dtype=np.int32),
        aod_track_row=np.asarray(aod_track_row, dtype=np.int32),
        collision_id=collisions,
        global_bc=bcs,
        bc_source=np.asarray(bc_source, dtype=np.int8),
        detector_mask=np.asarray(detector_mask, dtype=np.uint16),
        charge=np.asarray(charge, dtype=np.int8),
        pt_gev=np.asarray(pt_gev, dtype=np.float32),
        eta=np.asarray(eta, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        points_m=blender(np.concatenate(paths, axis=0) if paths else np.empty((0, 3), dtype=np.float32)),
        **extra,
    )
    print(f"Converted {len(ids)} trajectories to {prefix}_tracks.npz")
    root_file.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="single candidate ROOT file or quoted glob (e.g. 'events/candidate_*.root')")
    parser.add_argument("--output", default="", help="optional output prefix (single file) or directory (multiple files)")
    args = parser.parse_args()
    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No ROOT files matched {args.input!r}")
    for filename in files:
        path = Path(filename)
        prefix = (Path(args.output) if len(files) == 1 and args.output else
                  Path(args.output) / path.stem if args.output else path.with_suffix(""))
        convert(path, prefix)

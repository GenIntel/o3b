"""Train a TriplaneDistill feature field to preserve a teacher feature's vertex distances.

For random vertex pairs (i, j) of one instance the loss is

    (|g_i - g_j| - |f_i - f_j|)^2,

g the network's ``out_dim``-channel feature, f the teacher's per-vertex
feature exactly as the benchmark reads it: ``Mesh.load_or_convert`` of
``<geom>_vuni100_r512_f<teacher>`` (a concat is L2-normalised and weighted per
part), in the cached mesh frame the benchmark extracts features in. Pairs
never cross instances; consistency across instances comes from the shared
network, which only sees geometry.

Each step takes one mesh, samples a fresh surface point cloud (the only
augmentation), encodes it once and queries the field at the vertices of
``pairs`` random pairs. Every ``eval_every`` steps it reports, on held-out
pairs, the stress (sqrt of the mean squared distance error over the mean
squared teacher distance) and the Pearson correlation of the distances.

    python -m o3b.model.feat_distill.train --categories bread,facial_cream,hair_dryer,handbag,tooth_brush \\
        --teacher sd15+partfield --geometries fc16,fc32 --out checkpoints/feat_distill/pfd_sd15pf32.pth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch


def load_meshes(args) -> list[dict]:
    """[{oid, geom, verts (V,3) raw cached frame, faces, feats (V,F)}] for every
    object of the categories (the benchmark's filter: keypoints, synthetic)."""
    from o3b.data.datatypes.mesh import Mesh
    from o3b.dataset.dataset import build_dataset_from_config_or_name

    out = []
    for geom in args.geometries:
        mesh_type = f"{geom}_vuni100_r512_f{args.teacher}"
        ds = build_dataset_from_config_or_name({
            "dataset_name": "hc3d_object",
            "path_datasets_raw": args.path_datasets, "path_datasets_preprocess": args.path_datasets,
            "categories": args.categories, "item_type": "object", "mesh_type": mesh_type,
            "filter_has_kpts": True, "filter_is_real": False,
        })
        for i in ds._object_rows_id:
            row = ds._object_rows[i]
            oid = row["object_id"]
            mesh_rel = row.get("mesh_path")
            entry = ds.path_raw / mesh_rel if mesh_rel else ds.path_object_meshes / oid
            m = Mesh.load_or_convert(ds.path_preprocess / "mesh" / mesh_type / f"{oid}.glb", entry, mesh_type)
            out.append(dict(oid=oid, category=row["category"], geom=geom, verts=m.verts.float(),
                            faces=m.faces.long(), feats=m.vert_feats.float()))
            print(f"  {geom} {oid}: {len(m.verts)} verts, teacher {tuple(m.vert_feats.shape)}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", required=True, type=lambda s: s.split(","))
    p.add_argument("--teacher", default="sd15+partfield")
    p.add_argument("--geometries", default="fc16,fc32", type=lambda s: s.split(","))
    p.add_argument("--path-datasets", default="/work/dlclarge1/sommerl-od3d/datasets")
    p.add_argument("--out", required=True)
    p.add_argument("--out-dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--pairs", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-4, help="trunk (pretrained MeshFM)")
    p.add_argument("--lr-head", type=float, default=1e-3, help="the two new head layers")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--n-pc-points", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-meshfm-init", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda")
    from o3b.model.feat_distill.model import TriplaneDistillModel, normalize_verts
    from o3b.model.meshfm.meshfm.sampling import sample_surface_points

    print(f"loading teacher '{args.teacher}' on {args.geometries} for {args.categories}", flush=True)
    meshes = load_meshes(args)
    for m in meshes:
        m["vn"] = normalize_verts(m["verts"])
        m["feats"] = m["feats"].to(dev)
        # fixed held-out pairs per mesh for the report
        g = torch.Generator().manual_seed(hash(m["oid"] + m["geom"]) % 2**31)
        V = len(m["verts"])
        m["eval_pairs"] = torch.randint(0, V, (2, 4096), generator=g)
    print(f"{len(meshes)} meshes", flush=True)

    wrapper = TriplaneDistillModel(out_dim=args.out_dim, n_pc_points=args.n_pc_points)
    net = wrapper.build_network(dev, init_from_meshfm=not args.no_meshfm_init)
    net.train()
    head = [p_ for n, p_ in net.named_parameters()
            if n.startswith("triplane_transformer.upsampler") or n.startswith("triplane_transformer.mlp.4")]
    head_ids = {id(p_) for p_ in head}
    trunk = [p_ for p_ in net.parameters() if id(p_) not in head_ids]
    opt = torch.optim.AdamW([{"params": trunk, "lr": args.lr}, {"params": head, "lr": args.lr_head}],
                            weight_decay=1e-4)
    base_lrs = [args.lr, args.lr_head]

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    def field(m, gen=None):
        pc = sample_surface_points(m["vn"], m["faces"], args.n_pc_points, generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            planes = net(pc.unsqueeze(0).to(dev))
        return planes.float()

    def query(planes, m, idx):
        q = m["vn"][idx].clamp(-1, 1).unsqueeze(0).to(dev)
        return net.sample_triplane_features_from_points(planes, q)[0]

    @torch.no_grad()
    def evaluate():
        net.eval()
        stress, corr = [], []
        for m in meshes:
            planes = field(m, torch.Generator().manual_seed(0))
            i, j = m["eval_pairs"]
            g = query(planes, m, torch.cat([i, j]))
            dg = (g[: len(i)] - g[len(i):]).norm(dim=1)
            df = (m["feats"][i.to(dev)] - m["feats"][j.to(dev)]).norm(dim=1)
            stress.append(((dg - df).pow(2).mean() / df.pow(2).mean().clamp(min=1e-12)).sqrt().item())
            corr.append(torch.corrcoef(torch.stack([dg, df]))[0, 1].item())
        net.train()
        return sum(stress) / len(stress), sum(corr) / len(corr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()
    s, c = evaluate()
    print(f"step 0: held-out stress {s:.3f}, distance corr {c:.3f}", flush=True)
    hist.append(dict(step=0, stress=s, corr=c))
    run_loss = 0.0
    for step in range(args.steps):
        for gi, pg in enumerate(opt.param_groups):
            pg["lr"] = base_lrs[gi] * lr_scale(step)
        m = random.choice(meshes)
        planes = field(m)
        V = len(m["verts"])
        i = torch.randint(0, V, (args.pairs,))
        j = torch.randint(0, V, (args.pairs,))
        g = query(planes, m, torch.cat([i, j]))
        dg = (g[: args.pairs] - g[args.pairs:]).pow(2).sum(dim=1).add(1e-12).sqrt()
        with torch.no_grad():
            df = (m["feats"][i.to(dev)] - m["feats"][j.to(dev)]).norm(dim=1)
        loss = (dg - df).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: loss {run_loss / 50:.5f} ({time.time() - t0:.0f}s)", flush=True)
            run_loss = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            s, c = evaluate()
            print(f"step {step + 1}: held-out stress {s:.3f}, distance corr {c:.3f}", flush=True)
            hist.append(dict(step=step + 1, stress=s, corr=c))
            torch.save({"model": net.state_dict(), "args": vars(args), "history": hist}, out)
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

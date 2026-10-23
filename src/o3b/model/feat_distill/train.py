"""Train a TriplaneDistill feature field to preserve a teacher feature's vertex distances.

For random vertex pairs (i, j) of one instance the loss is

    (|g_i - g_j| - |f_i - f_j|)^2,

g the network's ``out_dim``-channel feature, f the teacher's per-vertex
feature exactly as the benchmark reads it: ``Mesh.load_or_convert`` of
``<geom>_vuni100_r512_f<teacher>`` (a concat is L2-normalised and weighted per
part), in the cached mesh frame the benchmark extracts features in. Pairs
never cross instances; consistency across instances comes from the shared
network, which only sees geometry.

Each step encodes ``meshes_per_step`` meshes, each from a fresh surface
point cloud (the only augmentation), and queries the field at the vertices of
random pairs, one loss over all of them.

``--cross-frac c`` > 0 adds pairs *across* instances: the meshes of a step come
in pairs (A, B) of one category and geometry, and a fraction c of each pair's
``pairs`` budget is (i on A, j on B) with target |f_A,i - f_B,j|, the rest
within A and within B. Within-instance pairs only fix each instance up to a
distance-preserving transform of its own feature space; the cross pairs ask
the instances' features to agree, which is what matching needs.

Every ``eval_every`` steps it reports, on fixed held-out pairs, the stress
(sqrt of the mean squared distance error over the mean squared teacher
distance) and the Pearson correlation of the distances, within instances and
across instances (each mesh against the next instance of its category and
geometry). ``--resume CKPT --steps 0`` only evaluates a checkpoint.

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
    """[{oid, geom, verts (V,3) raw cached frame, faces, feats (V,F)}] for the
    training instances of the categories.

    ``--instances eval`` (default): the benchmark's objects (keypoints,
    synthetic), i.e. training on the evaluated instances (transductive).
    ``--instances heldout:N``: N synthetic objects per category *without*
    keypoints, drawn with ``--seed``, so disjoint from the evaluated ones.
    """
    import random as _random

    from o3b.data.datatypes.mesh import Mesh
    from o3b.dataset.dataset import build_dataset_from_config_or_name

    heldout = args.instances.startswith("heldout:")
    out = []
    for geom in args.geometries:
        mesh_type = f"{geom}_vuni100_r512_f{args.teacher}"
        ds = build_dataset_from_config_or_name({
            "dataset_name": "hc3d_object",
            "path_datasets_raw": args.path_datasets, "path_datasets_preprocess": args.path_datasets,
            "categories": args.categories, "item_type": "object", "mesh_type": mesh_type,
            "filter_has_kpts": not heldout, "filter_is_real": False,
        })
        rows = [ds._object_rows[i] for i in ds._object_rows_id]
        if heldout:
            n = int(args.instances.split(":", 1)[1])
            picked = []
            for cat in args.categories:
                cand = sorted((r for r in rows if r["category"] == cat and r.get("obj_kpts3d") is None),
                              key=lambda r: r["object_id"])
                picked += _random.Random(args.seed).sample(cand, min(n, len(cand)))
            rows = picked
        for row in rows:
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
    p.add_argument("--geo-weight", type=float, default=0.0,
                   help="local geometry term on the within-instance pairs, d = |x_i - x_j| / S (S the "
                        "mesh's largest bbox side), c = --geo-clamp: (|g_i - g_j| - d)^2 for d < c; "
                        "for d >= c only a push apart, max(0, c - |g_i - g_j|)^2 (--geo-far hinge)")
    p.add_argument("--geo-far", choices=("hinge", "none"), default="hinge",
                   help="pairs beyond the clamp: 'hinge' pushes their features at least c apart, "
                        "'none' leaves them to the teacher term")
    p.add_argument("--geo-clamp", type=float, default=0.3)
    p.add_argument("--min-lr-frac", type=float, default=0.0,
                   help="cosine decays to this fraction of each base lr instead of to 0")
    p.add_argument("--n-pc-points", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-meshfm-init", action="store_true")
    p.add_argument("--meshes-per-step", type=int, default=1)
    p.add_argument("--cross-frac", type=float, default=0.0,
                   help="fraction of a mesh pair's pairs that cross the two instances (0: within only)")
    p.add_argument("--resume", default=None, help="state dict to start from (with --steps 0: evaluate only)")
    p.add_argument("--instances", default="eval",
                   help="'eval': the benchmark's (evaluated) objects; 'heldout:N': N objects per category "
                        "without keypoints, disjoint from the evaluated ones")
    p.add_argument("--extract-only", action="store_true",
                   help="only load the training meshes (extracting and caching the teacher), then exit")
    p.add_argument("--init-ckpt", default=None,
                   help="plain state dict to initialise the trunk from instead of MeshFM, e.g. PartField's "
                        "(checkpoints/partfield/model_objaverse_trunk_state_dict.pth)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda")
    from o3b.model.feat_distill.model import TriplaneDistillModel, normalize_verts
    from o3b.model.meshfm.meshfm.sampling import sample_surface_points

    print(f"loading teacher '{args.teacher}' on {args.geometries} for {args.categories}", flush=True)
    meshes = load_meshes(args)
    if args.extract_only:
        print(f"extracted {len(meshes)} meshes: {sorted({m['oid'] for m in meshes})}", flush=True)
        return
    for m in meshes:
        m["vn"] = normalize_verts(m["verts"])
        m["size"] = float((m["verts"].amax(0) - m["verts"].amin(0)).max())   # largest bbox side, as the PCK
        m["verts_dev"] = m["verts"].to(dev)
        m["feats"] = m["feats"].to(dev)
        # fixed held-out pairs per mesh for the report
        g = torch.Generator().manual_seed(hash(m["oid"] + m["geom"]) % 2**31)
        V = len(m["verts"])
        m["eval_pairs"] = torch.randint(0, V, (2, 4096), generator=g)
    groups = {}
    for k, m in enumerate(meshes):
        groups.setdefault((m["category"], m["geom"]), []).append(k)
    for grp in groups.values():                      # held-out cross pairs: each mesh vs the next instance
        for a, b in zip(grp, grp[1:] + grp[:1]):
            g = torch.Generator().manual_seed(hash(meshes[a]["oid"] + meshes[b]["oid"]) % 2**31)
            meshes[a]["eval_partner"] = b
            meshes[a]["eval_cross"] = torch.stack([torch.randint(0, len(meshes[a]["verts"]), (4096,), generator=g),
                                                   torch.randint(0, len(meshes[b]["verts"]), (4096,), generator=g)])
    if args.cross_frac > 0 and args.meshes_per_step % 2:
        raise ValueError("--cross-frac needs an even --meshes-per-step (meshes come in instance pairs)")
    print(f"{len(meshes)} meshes, {len(groups)} category x geometry groups", flush=True)

    wrapper = TriplaneDistillModel(out_dim=args.out_dim, n_pc_points=args.n_pc_points)
    net = wrapper.build_network(dev, init_from_meshfm=not args.no_meshfm_init and not args.resume,
                                init_state_path=None if args.resume else args.init_ckpt)
    print(f"initialised {len(net.init_loaded)} / {len(net.state_dict())} tensors from "
          f"{args.init_ckpt or ('MeshFM' if net.init_loaded else 'nothing')}", flush=True)
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"] if "model" in st else st)
    net.train()
    # parameters the initialisation did not provide get the head learning rate
    head = [p_ for n, p_ in net.named_parameters() if net.init_loaded and n not in net.init_loaded] or \
        [p_ for n, p_ in net.named_parameters()
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
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))

    def field(m, gen=None):
        pc = sample_surface_points(m["vn"], m["faces"], args.n_pc_points, generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            planes = net(pc.unsqueeze(0).to(dev))
        return planes.float()

    def query(planes, m, idx):
        q = m["vn"][idx].clamp(-1, 1).unsqueeze(0).to(dev)
        return net.sample_triplane_features_from_points(planes, q)[0]

    def stress_corr(dg, df):
        return (((dg - df).pow(2).mean() / df.pow(2).mean().clamp(min=1e-12)).sqrt().item(),
                torch.corrcoef(torch.stack([dg, df]))[0, 1].item())

    @torch.no_grad()
    def evaluate():
        """(within stress, within corr, cross stress, cross corr), means over meshes."""
        net.eval()
        feats_at = {}
        for k, m in enumerate(meshes):
            planes = field(m, torch.Generator().manual_seed(0))
            idx = torch.unique(torch.cat([m["eval_pairs"].flatten()]
                                         + [meshes[a]["eval_cross"][1] for a in range(len(meshes))
                                            if meshes[a]["eval_partner"] == k]
                                         + [m["eval_cross"][0]]))
            feats_at[k] = (idx, query(planes, m, idx))
        def g_of(k, ids):
            idx, g = feats_at[k]
            return g[torch.searchsorted(idx, ids)]
        w, x = [], []
        for k, m in enumerate(meshes):
            i, j = m["eval_pairs"]
            w.append(stress_corr((g_of(k, i) - g_of(k, j)).norm(dim=1),
                                 (m["feats"][i.to(dev)] - m["feats"][j.to(dev)]).norm(dim=1)))
            b = m["eval_partner"]
            i, j = m["eval_cross"]
            x.append(stress_corr((g_of(k, i) - g_of(b, j)).norm(dim=1),
                                 (m["feats"][i.to(dev)] - meshes[b]["feats"][j.to(dev)]).norm(dim=1)))
        net.train()
        mean = lambda v, c: sum(t[c] for t in v) / len(v)
        return mean(w, 0), mean(w, 1), mean(x, 0), mean(x, 1)

    def report(step):
        ws, wc, xs, xc = evaluate()
        print(f"step {step}: held-out within stress {ws:.3f} corr {wc:.3f} | across instances stress {xs:.3f} "
              f"corr {xc:.3f}", flush=True)
        hist.append(dict(step=step, stress=ws, corr=wc, cross_stress=xs, cross_corr=xc))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()
    report(0)
    if args.steps == 0:
        out.with_suffix(".eval.json").write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
        return

    def pair_loss(ga, gb, fa, fb):
        dg = (ga - gb).pow(2).sum(dim=1).add(1e-12).sqrt()
        with torch.no_grad():
            df = (fa - fb).norm(dim=1)
        return (dg - df).pow(2)

    def within(m, planes, n):
        i, j = torch.randint(0, len(m["verts"]), (2, n))
        g = query(planes, m, torch.cat([i, j]))
        loss = pair_loss(g[:n], g[n:], m["feats"][i.to(dev)], m["feats"][j.to(dev)])
        if args.geo_weight > 0:
            # local geometry on the same pairs: closer than the clamp c, the feature
            # distance should equal the 3-D distance in units of the mesh size;
            # beyond it, features closer than c are only pushed apart to c
            c = args.geo_clamp
            dg = (g[:n] - g[n:]).pow(2).sum(dim=1).add(1e-12).sqrt()
            with torch.no_grad():
                dx = (m["verts_dev"][i.to(dev)] - m["verts_dev"][j.to(dev)]).norm(dim=1) / m["size"]
                near = dx < c
            geo = torch.where(near, (dg - dx).pow(2),
                              (c - dg).clamp(min=0).pow(2) if args.geo_far == "hinge" else torch.zeros_like(dg))
            loss = loss + args.geo_weight * geo
        return loss

    run_loss = 0.0
    for step in range(args.steps):
        for gi, pg in enumerate(opt.param_groups):
            pg["lr"] = base_lrs[gi] * lr_scale(step)
        # one backward per unit (an instance pair, or one mesh), gradients
        # accumulated: every unit has the same number of pairs, so this equals
        # the mean over all pairs of the step, at one unit's activation memory
        opt.zero_grad(set_to_none=True)
        n_units = args.meshes_per_step // 2 if args.cross_frac > 0 else args.meshes_per_step
        loss_val = 0.0
        for _ in range(n_units):
            if args.cross_frac > 0:
                a = random.randrange(len(meshes))
                grp = groups[(meshes[a]["category"], meshes[a]["geom"])]
                b = random.choice([k for k in grp if k != a])
                A, B = meshes[a], meshes[b]
                pa, pb = field(A), field(B)
                n_x = int(round(args.pairs * args.cross_frac))
                n_w = (args.pairs - n_x) // 2
                i = torch.randint(0, len(A["verts"]), (n_x,))
                j = torch.randint(0, len(B["verts"]), (n_x,))
                terms = [within(A, pa, n_w), within(B, pb, n_w),
                         pair_loss(query(pa, A, i), query(pb, B, j), A["feats"][i.to(dev)], B["feats"][j.to(dev)])]
            else:
                m = random.choice(meshes)
                terms = [within(m, field(m), args.pairs)]
            unit_loss = torch.cat(terms).mean() / n_units
            unit_loss.backward()
            loss_val += unit_loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run_loss += loss_val
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: loss {run_loss / 50:.5f} ({time.time() - t0:.0f}s)", flush=True)
            run_loss = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            report(step + 1)
            torch.save({"model": net.state_dict(), "args": vars(args), "history": hist}, out)
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

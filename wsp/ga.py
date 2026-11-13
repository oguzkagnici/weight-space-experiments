"""Genetic-loop driver.

  ga init
  ga new-gen --editors claude,ed_a,ed_b --budget 6 --max-cands 2 [--sel-n 800]
  ga candidates                     list the current generation's submitted candidates
  ga select [--wait]                evaluate champion + every candidate on this generation's hidden selection subset
  ga decide [--z 2.0] [--force CAND|--keep]   choose the winner (paired gain vs champion, needs z >= threshold)
  ga heldout [WHO] [--n 1000]       evaluate a champion ('champion', 'base' or a champion id) on the fixed held-out subset
  ga status

Layout: runs/ga_state.json, runs/gen_NNN/{gen.json, editors/<name>/..., selection/}, runs/champions/<id>/weights/"""
import argparse, json, math, os, random, shutil, sys, time
import numpy as np
from safetensors.torch import load_file, save_file
import torch

from .config import RUNS, QUEUE, ROOT
from .server import locked_update, summarize, paired
from . import data

STATE = os.path.join(RUNS, "ga_state.json")
PY = "/home/user/venv/bin/python"


def state():
    return json.load(open(STATE))


def save_state(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)


def gen_dir(g):
    return os.path.join(RUNS, f"gen_{g:03d}")


def enqueue_eval(jid, weights, qids, out, priority=0, protocol="direct", cap=None):
    job = {"id": jid, "type": "eval_gen" if protocol == "think" else "eval", "priority": priority,
           "created": time.time(), "weights": weights, "questions": qids, "out": out}
    if cap:
        job["cap"] = cap   # thinking-mode token cap other than gen.THINK_CAP (D60)
    p = os.path.join(QUEUE, "pending", jid + ".json")
    json.dump(job, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)


def wait_jobs(jids, poll=10, timeout=None):
    t0 = time.time(); res = {}
    while True:
        for j in jids:
            if j in res: continue
            for d in ("done", "failed"):
                p = os.path.join(QUEUE, d, j + ".json")
                if os.path.exists(p):
                    res[j] = json.load(open(p))
        if len(res) == len(jids) or (timeout and time.time() - t0 > timeout):
            return res
        time.sleep(poll)


def load_rows(path):
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------- commands
def cmd_init(a):
    os.makedirs(os.path.join(RUNS, "champions"), exist_ok=True)
    if os.path.exists(STATE) and not a.force:
        print("already initialised"); return
    save_state({"gen": 0, "champion": "base", "champion_weights": None, "lineage": [], "history": []})
    print("initialised: champion = base LFM3-1.7B")


def write_wrappers(edir):
    for name, mod in (("py", ""), ("fwd", "-m wsp.fwd")):
        p = os.path.join(edir, name)
        open(p, "w").write(f'#!/bin/bash\ncd "{edir}" && WSP_EDITOR_DIR="{edir}" PYTHONPATH="{ROOT}" '
                           f'exec {PY} {mod} "$@"\n')
        os.chmod(p, 0o755)


def cmd_new_gen(a):
    st = state()
    g = st["gen"] + 1
    gd = gen_dir(g)
    if os.path.exists(gd):
        sys.exit(f"{gd} exists")
    sel_pool = data.splits()["selection"]
    if a.sel_from:   # D68: reuse an earlier generation's selection questions (same parent -> its cached answers are reused)
        sel = json.load(open(os.path.join(gen_dir(a.sel_from), "gen.json")))["selection_questions"]
        a.sel_n = len(sel)
    else:
        sel = sorted(random.Random(f"selection-gen-{g}").sample(sel_pool, a.sel_n))
    os.makedirs(os.path.join(gd, "editors")); os.makedirs(os.path.join(gd, "selection"))
    gen = {"gen": g, "name": f"gen_{g:03d}", "parent": st["champion"], "parent_weights": st["champion_weights"],
           "editors": a.editors.split(","), "budget": a.budget, "max_candidates": a.max_cands,
           "selection_questions": sel, "created": time.time(), "focus": json.loads(a.focus) if a.focus else {},
           "protocol": a.protocol, "template": a.template, "sel_cap": a.sel_cap, "sel_from": a.sel_from}
    if a.trace_bank:
        tb = os.path.abspath(a.trace_bank)
        gen["trace_bank"] = tb; gen["trace_n"] = sum(1 for _ in open(tb))
    json.dump(gen, open(os.path.join(gd, "gen.json"), "w"), indent=1)
    for ed in gen["editors"]:
        edir = os.path.join(gd, "editors", ed)
        for sub in ("edited", "work", "code", "candidates"):
            os.makedirs(os.path.join(edir, sub))
        json.dump({"editor": ed, "gen": g, "gen_name": gen["name"], "parent": st["champion"],
                   "parent_weights": st["champion_weights"], "budget": a.budget, "max_candidates": a.max_cands},
                  open(os.path.join(edir, "meta.json"), "w"), indent=1)
        json.dump({"passes_used": 0, "pending": []}, open(os.path.join(edir, "state.json"), "w"))
        open(os.path.join(edir, "notes.md"), "w").write(f"# Notes of editor {ed}, generation {g}\n\n")
        write_wrappers(edir)
    st["gen"] = g; save_state(st)
    print(f"generation {g} created at {gd}; parent = {st['champion']}; editors = {gen['editors']}; "
          f"budget {a.budget} passes each; selection subset {a.sel_n} hidden questions")


def candidates(g):
    out = []
    ed_root = os.path.join(gen_dir(g), "editors")
    for ed in sorted(os.listdir(ed_root)):
        cdir = os.path.join(ed_root, ed, "candidates")
        for c in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
            p = os.path.join(cdir, c)
            if os.path.exists(os.path.join(p, "manifest.json")):
                out.append({"name": f"{ed}/{c}", "dir": p, "manifest": json.load(open(os.path.join(p, "manifest.json"))),
                            "description": open(os.path.join(p, "description.md")).read()})
    return out


def cmd_candidates(a):
    g = state()["gen"]
    for c in candidates(g):
        tot = sum(v["changed_bf16"] for v in c["manifest"]["tensors"].values())
        print(f"- {c['name']}: {len(c['manifest']['tensors'])} tensors, {tot} weights\n    {c['description'][:300]}")


def cmd_select(a):
    st = state(); g = st["gen"]; gd = gen_dir(g)
    gen = json.load(open(os.path.join(gd, "gen.json")))
    sel = gen["selection_questions"]
    parent = [d for d in [gen["parent_weights"]] if d]
    cap = gen.get("sel_cap")
    if a.parent_chunks:
        # D65: run the parent's side early in small jobs (per-item cache), so editor passes can run in between
        os.makedirs(os.path.join(gd, "selection", "prerun"), exist_ok=True)
        for i in range(0, len(sel), a.parent_chunks):
            jid = f"{gen['name']}_prerun_parent_{i // a.parent_chunks:02d}"
            enqueue_eval(jid, parent, sel[i:i + a.parent_chunks], os.path.join(gd, "selection", "prerun", jid + ".jsonl"),
                         priority=a.priority, protocol=gen.get("protocol", "direct"), cap=cap)
        print(f"enqueued the parent's {len(sel)} selection answers in chunks of {a.parent_chunks} (cap {cap})"); return
    jobs = {}
    jid = f"{gen['name']}_sel_champion"
    out = os.path.join(gd, "selection", "champion.jsonl")
    if not os.path.exists(out):
        enqueue_eval(jid, parent, sel, out, priority=a.priority, protocol=gen.get("protocol", "direct"), cap=cap); jobs["champion"] = jid
    for c in candidates(g):
        safe = c["name"].replace("/", "__")
        out = os.path.join(gd, "selection", f"{safe}.jsonl")
        if os.path.exists(out): continue
        jid = f"{gen['name']}_sel_{safe}"
        enqueue_eval(jid, parent + [os.path.join(c["dir"], "edited")], sel, out, priority=a.priority, protocol=gen.get("protocol", "direct"), cap=cap)
        jobs[c["name"]] = jid
    print(f"enqueued {len(jobs)} selection evaluations on {len(sel)} hidden questions: {list(jobs)}")
    if a.wait and jobs:
        res = wait_jobs(list(jobs.values()))
        for k, j in jobs.items():
            r = res[j]; print(k, "ERROR " + r["error"] if "error" in r else f"acc {r['result']['acc']} ({r['seconds']}s)")


def gen_protocol(g):
    return json.load(open(os.path.join(gen_dir(g), "gen.json"))).get("protocol", "direct")


def selection_table(g):
    gd = gen_dir(g)
    champ_p = os.path.join(gd, "selection", "champion.jsonl")
    if not os.path.exists(champ_p):
        return None, []
    ch = load_rows(champ_p)
    table = []
    if gen_protocol(g) == "think":
        from .gen import summarize_gen
        champ_summary = summarize_gen(ch)
    else:
        champ_summary = summarize(ch)
    for c in candidates(g):
        p = os.path.join(gd, "selection", c["name"].replace("/", "__") + ".jsonl")
        if not os.path.exists(p):
            table.append({"name": c["name"], "status": "not evaluated"}); continue
        rows = load_rows(p)
        if gen_protocol(g) == "think":
            from .gen import paired_gen, summarize_gen
            pr = paired_gen(ch, rows); s = summarize_gen(rows); pr.setdefault("logp_correct_diff", 0.0)
        else:
            pr = paired(ch, rows); s = summarize(rows)
        z = pr["acc_diff"] / pr["acc_diff_se"] if pr["acc_diff_se"] else 0.0
        bf = both_finish(ch, rows) if gen_protocol(g) == "think" else {}
        gd_ = guard_result(g, c["name"])
        table.append({"name": c["name"], "acc": s["acc"], **pr, "z": round(z, 2), **bf, **gd_,
                      "weights_changed": sum(v["changed_bf16"] for v in c["manifest"]["tensors"].values()),
                      "description": c["description"]})
    return champ_summary, table


def both_finish(parent_rows, cand_rows):
    """D65: paired accuracy on the questions that neither model cut at the cap (content, not efficiency)."""
    o = {(r["id"], r["sample"]): r for r in parent_rows}
    d = [r["correct"] - o[(r["id"], r["sample"])]["correct"] for r in cand_rows
         if (r["id"], r["sample"]) in o and not r["cut"] and not o[(r["id"], r["sample"])]["cut"]]
    n = len(d)
    if n < 2:
        return {"bf_n": n}
    m = sum(d) / n; se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return {"bf_n": n, "bf_diff": round(100 * m, 2), "bf_se": round(100 * se, 2), "bf_z": round(m / se, 2) if se else 0.0,
            "bf_plus": sum(x > 0 for x in d), "bf_minus": sum(x < 0 for x in d)}


GUARD = "gsm8k_boxed"   # D65 transfer guard: different task, different answer format, cap 8,192


def guard_path(g, who):
    return os.path.join(gen_dir(g), "guard", who.replace("/", "__") + f"_{GUARD}_cap8192.jsonl")


def cmd_guard(a):
    """Queue the D65 transfer guard (GSM8K with a \\boxed{} answer, 250 items, cap 8,192) for the parent and every candidate."""
    g = state()["gen"]; gen = json.load(open(os.path.join(gen_dir(g), "gen.json")))
    from .bench import BDIR
    items_file = os.path.join(BDIR, GUARD, "items.jsonl")
    parent = [d for d in [gen["parent_weights"]] if d]
    whos = [("parent", parent)] + [(c["name"], parent + [os.path.join(c["dir"], "edited")]) for c in candidates(g)]
    os.makedirs(os.path.join(gen_dir(g), "guard"), exist_ok=True)
    for who, w in whos:
        out = guard_path(g, who)
        if os.path.exists(out): continue
        jid = f"{gen['name']}_guard_{who.replace('/', '__')}"
        job = {"id": jid, "type": "bench", "bench": GUARD, "priority": a.priority, "created": time.time(), "weights": w,
               "items_file": items_file, "cap": 8192, "room": 1024, "out": out}
        p = os.path.join(QUEUE, "pending", jid + ".json"); json.dump(job, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        print("queued", jid)


def guard_result(g, name):
    pp, cp = guard_path(g, "parent"), guard_path(g, name)
    if not (os.path.exists(pp) and os.path.exists(cp)):
        return {}
    from .bench import BDIR, score, _paired
    items = [json.loads(l) for l in open(os.path.join(BDIR, GUARD, "items.jsonl"))]
    sp = score(GUARD, {r["id"]: r for r in load_rows(pp)}, items)
    sc = score(GUARD, {r["id"]: r for r in load_rows(cp)}, items)
    pr = _paired(sp, sc)
    return {"guard_diff": pr["diff"], "guard_se": pr["se"], "guard_z": pr["z"]}


def cmd_decide(a):
    st = state(); g = st["gen"]; gd = gen_dir(g)
    champ, table = selection_table(g)
    if champ is None:
        sys.exit("run `ga select --wait` first")
    ev = [t for t in table if "acc_diff" in t]
    ev.sort(key=lambda t: (t["acc_diff"], t["logp_correct_diff"]), reverse=True)
    print(f"champion ({st['champion']}): acc {champ['acc']} on {champ['n']} hidden selection questions")
    for t in ev:
        print(f"  {t['name']:40s} acc {t['acc']:6.2f}  diff {t['acc_diff']:+6.2f} (SE {t['acc_diff_se']}, z {t['z']:+.2f})"
              f"  +{t['wrong_to_right']}/-{t['right_to_wrong']}  dlogp {t['logp_correct_diff']:+.4f}  w={t['weights_changed']}")
        if "bf_diff" in t:
            print(f"      both finished: n {t['bf_n']}, diff {t['bf_diff']:+.2f} (SE {t['bf_se']}, z {t['bf_z']:+.2f}), "
                  f"+{t['bf_plus']}/-{t['bf_minus']}")
        if "guard_diff" in t:
            print(f"      guard {GUARD}: diff {t['guard_diff']:+.2f} (SE {t['guard_se']}, z {t['guard_z']})")
    def ok(t):
        if a.bf_z is not None and not (t.get("bf_z") is not None and t["bf_z"] >= a.bf_z and t.get("bf_diff", 0) > 0):
            return False
        if a.guard_z is not None and not ("guard_z" in t and (t["guard_z"] or 0) > a.guard_z):
            return False
        return True
    winner = None
    if a.force:
        winner = next(t for t in ev if t["name"] == a.force)
    elif not a.keep:
        winner = next((t for t in ev if t["z"] >= a.z and t["acc_diff"] > 0 and ok(t)), None)
    rec = {"gen": g, "parent": st["champion"], "champion_selection": champ, "candidates": table,
           "winner": winner["name"] if winner else None,
           "rule": f"z >= {a.z}" + (f"; both-finish z >= {a.bf_z}" if a.bf_z is not None else "")
                   + (f"; {GUARD} guard z > {a.guard_z}" if a.guard_z is not None else ""), "time": time.time()}
    if winner:
        cid = f"gen{g:03d}_{winner['name'].replace('/', '_')}"
        cdir = os.path.join(RUNS, "champions", cid, "weights"); os.makedirs(cdir)
        if st["champion_weights"]:
            for fn in os.listdir(st["champion_weights"]):
                shutil.copy2(os.path.join(st["champion_weights"], fn), cdir)
        cand = next(c for c in candidates(g) if c["name"] == winner["name"])
        for fn in os.listdir(os.path.join(cand["dir"], "edited")):
            shutil.copy2(os.path.join(cand["dir"], "edited", fn), cdir)
        json.dump({"id": cid, "gen": g, "from": winner["name"], "parent": st["champion"],
                   "description": winner["description"], "selection": {k: winner[k] for k in
                   ("acc", "acc_diff", "acc_diff_se", "z", "wrong_to_right", "right_to_wrong", "logp_correct_diff")}},
                  open(os.path.join(RUNS, "champions", cid, "info.json"), "w"), indent=1)
        st["lineage"].append(cid); st["champion"] = cid; st["champion_weights"] = cdir
        print(f"WINNER: {winner['name']} -> new champion {cid}")
    else:
        print("no candidate passed the rule: the champion survives (elitism)")
    st["history"].append({k: rec[k] for k in ("gen", "parent", "winner", "rule")} | {
        "champion_acc": champ["acc"], "candidates": [{k: t.get(k) for k in ("name", "acc", "acc_diff", "acc_diff_se", "z",
        "wrong_to_right", "right_to_wrong", "logp_correct_diff", "weights_changed", "description", "bf_n", "bf_diff", "bf_se",
        "bf_z", "bf_plus", "bf_minus", "guard_diff", "guard_se", "guard_z")} for t in table]})
    save_state(st)
    json.dump(rec, open(os.path.join(gd, "decision.json"), "w"), indent=1)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "ledger.jsonl"), "a") as f:
        f.write(json.dumps({"gen": g, "parent": rec["parent"], "winner": rec["winner"], "rule": rec["rule"],
                            "parent_selection_acc": champ["acc"], "selection_n": champ["n"], "new_champion": st["champion"],
                            "candidates": [{k: t.get(k) for k in ("name", "acc", "acc_diff", "acc_diff_se", "z",
                                            "wrong_to_right", "right_to_wrong", "logp_correct_diff", "weights_changed",
                                            "bf_n", "bf_diff", "bf_se", "bf_z", "guard_diff", "guard_z")}
                                           for t in table], "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


def heldout_ids(n):
    ids = sorted(data.splits()["heldout"]); random.Random("heldout-fixed").shuffle(ids)
    return ids[:n]


def cmd_heldout(a):
    st = state()
    who = a.who
    if who == "champion": who = st["champion"]
    weights = [] if who == "base" else [os.path.join(RUNS, "champions", who, "weights")]
    proto = getattr(a, "protocol", "direct")
    if a.set == "gpqa":
        qids = data.gpqa_ids(); a.n = "gpqa"
        if not qids: sys.exit("GPQA not built: HF_TOKEN=... python -c 'from wsp.data import build_gpqa; build_gpqa()'")
    else:
        qids = heldout_ids(a.n)
    os.makedirs(os.path.join(RUNS, "heldout"), exist_ok=True)
    cap = getattr(a, "cap", None); csuf = f"_cap{cap}" if cap else ""
    out = os.path.join(RUNS, "heldout", f"{who}_n{a.n}{'_think' if proto == 'think' else ''}{csuf}.jsonl")
    if not os.path.exists(out):
        jid = f"heldout_{who}_n{a.n}"
        jid += ("_think" if proto == "think" else "") + csuf
        enqueue_eval(jid, weights, qids, out, priority=1, protocol=proto, cap=cap)
        r = wait_jobs([jid])[jid]
        if "error" in r: sys.exit(r["error"])
    rows = load_rows(out)
    suffix = ("_think" if proto == "think" else "") + csuf
    base_out = os.path.join(RUNS, "heldout", f"base_n{a.n}{suffix}.jsonl")
    if proto == "think":
        from .gen import summarize_gen, paired_gen
        s = summarize_gen(rows)
        print(f"{who}: held-out thinking-mode {json.dumps(s)}")
        if who != "base" and os.path.exists(base_out):
            print("paired vs base:", paired_gen(load_rows(base_out), rows))
        return
    s = summarize(rows)
    print(f"{who}: held-out acc {s['acc']} (SE {s['se']}) on {s['n']}")
    print("by category:", {c: v for c, v in s["by_category"].items()})
    if who != "base" and os.path.exists(base_out):
        print("paired vs base:", paired(load_rows(base_out), rows))


def history_text(st, upto_gen):
    parts = ["HISTORY OF THE LOOP."]
    if not st["lineage"]:
        parts.append("No edit has been accepted yet: the parent is the unedited LFM3-1.7B.")
    else:
        parts.append("Accepted edits so far (the parent contains all of them, in this order):")
        for cid in st["lineage"]:
            info = json.load(open(os.path.join(RUNS, "champions", cid, "info.json")))
            s = info["selection"]
            parts.append(f"- {cid} (from {info['from']}): hidden-selection gain {s['acc_diff']:+.2f} points "
                         f"(SE {s['acc_diff_se']}, +{s['wrong_to_right']}/-{s['right_to_wrong']}). Description: "
                         f"{info['description'][:1500]}")
    prev = [h for h in st["history"] if h["gen"] < upto_gen]
    for h in prev[-2:]:
        parts.append(f"\nGeneration {h['gen']} (parent {h['parent']}, parent accuracy {h['champion_acc']} on its hidden "
                     f"selection questions). Winner: {h['winner'] or 'none, the parent survived'}. Candidates:")
        for c in h["candidates"]:
            if c.get("acc_diff") is None: continue
            parts.append(f"- {c['name']}: {c['acc_diff']:+.2f} points (SE {c['acc_diff_se']}, z {c['z']:+.2f}), "
                         f"{c['weights_changed']} weights. {c['description'][:700]}")
        rep_root = os.path.join(gen_dir(h["gen"]), "editors")
        for ed in sorted(os.listdir(rep_root)):
            rp = os.path.join(rep_root, ed, "report.md"); npth = os.path.join(rep_root, ed, "notes.md")
            if os.path.exists(rp):
                parts.append(f"\nReport of editor {ed} (generation {h['gen']}):\n{open(rp).read()[:3500]}")
            elif ed != "carry" and os.path.exists(npth):
                # sub-agents sometimes cannot create report.md: use a '## REPORT' section of notes.md, else its end
                nt = open(npth).read(); i = nt.rfind("## REPORT")
                body = nt[i:][:3500] if i >= 0 else nt[-3500:]
                parts.append(f"\nReport of editor {ed} (generation {h['gen']}; no report.md, so "
                             f"{'the REPORT section' if i >= 0 else 'the end'} of its notes.md):\n{body}")
    return "\n".join(parts)


def cmd_brief(a):
    st = state(); g = st["gen"]; gd = gen_dir(g)
    gen = json.load(open(os.path.join(gd, "gen.json")))
    tpl = open(os.path.join(ROOT, "prompts", gen.get("template", "editor_task.md"))).read()
    eds = [a.editor] if a.editor else gen["editors"]
    for ed in eds:
        edir = os.path.join(gd, "editors", ed)
        note = ("It is the unedited model." if not st["lineage"] else
                "It already contains the accepted edits listed in the history below; load_base() gives the unedited model.")
        txt = (tpl.replace("{EDITOR}", ed).replace("{GEN}", str(g)).replace("{MAX_CANDS}", str(gen["max_candidates"]))
               .replace("{PARENT}", st["champion"]).replace("{PARENT_NOTE}", note).replace("{EDIR}", edir)
               .replace("{BUDGET}", str(gen["budget"])).replace("{HISTORY}", history_text(st, g))
               .replace("{TRACE_BANK}", str(gen.get("trace_bank", ""))).replace("{TRACE_N}", str(gen.get("trace_n", ""))))
        if gen.get("focus", {}).get(ed):
            txt += ("\n\nSUGGESTED STARTING POINT (to diversify the generation; abandon it if the evidence points "
                    "elsewhere): " + gen["focus"][ed])
        open(os.path.join(edir, "TASK.md"), "w").write(txt)
        print(f"wrote {edir}/TASK.md ({len(txt)} chars)")


def cmd_report(a):
    st = state(); g = a.gen or st["gen"]; gd = gen_dir(g)
    gen = json.load(open(os.path.join(gd, "gen.json")))
    dec = json.load(open(os.path.join(gd, "decision.json"))) if os.path.exists(os.path.join(gd, "decision.json")) else None
    L = [f"# Generation {g} report", "",
         f"- Created: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(gen['created']))}",
         f"- Parent: `{gen['parent']}`",
         f"- Editors: {', '.join(gen['editors'])}; budget {gen['budget']} passes each; up to {gen['max_candidates']} candidates each",
         f"- Hidden selection subset: {len(gen['selection_questions'])} questions (seed `selection-gen-{g}`)", ""]
    if gen.get("focus"):
        L += ["## Suggested starting points", ""] + [f"- **{e}**: {t}" for e, t in gen["focus"].items()] + [""]
    L += ["## Forward passes", "", "| editor | pass | type | model | n | headline | seconds |", "|---|---|---|---|---|---|---|"]
    for ed in gen["editors"]:
        pp = os.path.join(gd, "editors", ed, "passes.jsonl")
        for line in open(pp) if os.path.exists(pp) else []:
            r = json.loads(line); res = r.get("result") or {}
            if r["type"] == "score":
                pr = res.get("paired_edited_minus_original")
                m = res.get("edited") or res.get("original") or {}
                head = (f"paired {pr['acc_diff']:+.2f} (SE {pr['acc_diff_se']}), +{pr['wrong_to_right']}/-{pr['right_to_wrong']}, "
                        f"dlogp {pr['logp_correct_diff']:+.4f}" if pr else f"acc {m.get('acc')}")
                nq = m.get("n")
            else:
                head = f"read {res.get('texts')} texts"; nq = res.get("texts")
            acts = " +acts" if (r.get("request") or {}).get("activations") else ""
            L.append(f"| {ed} | {r['k']} | {r['type']}{acts} | {r.get('model')} | {nq} | {head} | {r.get('seconds')} |")
    L += ["", "## Candidates", ""]
    for c in candidates(g):
        tot = sum(v["changed_bf16"] for v in c["manifest"]["tensors"].values())
        L += [f"### {c['name']}", "", f"- Tensors: {', '.join(c['manifest']['tensors'])}", f"- Weights changed (bf16): {tot}",
              "", c["description"].strip(), ""]
    if dec:
        ch = dec["champion_selection"]
        L += ["## Selection (hidden questions) and decision", "",
              f"Parent accuracy on the {ch['n']} selection questions: **{ch['acc']}%**. Rule: {dec['rule']}.", "",
              "| candidate | acc | paired diff | SE | z | wrong->right | right->wrong | d logp(correct) | weights |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in sorted([t for t in dec["candidates"] if "acc_diff" in t], key=lambda t: -t["acc_diff"]):
            L.append(f"| {t['name']} | {t['acc']} | {t['acc_diff']:+.2f} | {t['acc_diff_se']} | {t['z']:+.2f} | "
                     f"{t['wrong_to_right']} | {t['right_to_wrong']} | {t['logp_correct_diff']:+.4f} | {t['weights_changed']} |")
        L += ["", f"**Decision:** {'winner `' + dec['winner'] + '` becomes the new champion' if dec['winner'] else 'no candidate passed; the parent survives'}.", ""]
    L += ["## Editors' reports", ""]
    for ed in gen["editors"]:
        rp = os.path.join(gd, "editors", ed, "report.md")
        L += [f"### {ed}", "", open(rp).read().strip() if os.path.exists(rp) else "_(no report yet)_", ""]
    out = os.path.join(gd, "REPORT.md"); open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)


def cmd_crossover(a):
    """Merge candidates by summing their weight deltas relative to the generation's parent, then stage the result in an
    editor's edited/ folder (that editor still measures and submits it). Columns edited by several parents get the
    sum of their deltas; overlaps are reported."""
    st = state(); g = a.gen or st["gen"]
    gen = json.load(open(os.path.join(gen_dir(g), "gen.json")))
    def parent_tensor(name):
        p = gen["parent_weights"] and os.path.join(gen["parent_weights"], name + ".safetensors")
        if p and os.path.exists(p): return load_file(p)["tensor"].float()
        from safetensors import safe_open
        from .config import WEIGHTS_FILE
        with safe_open(WEIGHTS_FILE, "pt") as f: return f.get_tensor(name).float()
    cands = {c["name"]: c for c in candidates(g)}
    deltas = {}
    for nm in a.cands.split(","):
        d = os.path.join(cands[nm]["dir"], "edited")
        for fn in os.listdir(d):
            name = fn[:-12]; t = load_file(os.path.join(d, fn))["tensor"].float()
            delta = t - parent_tensor(name)
            if name in deltas:
                ov = ((deltas[name] != 0) & (delta != 0)).any(0).sum().item() if delta.dim() == 2 else 0
                print(f"overlap in {name}: {ov} columns edited by more than one parent")
                deltas[name] = deltas[name] + delta
            else:
                deltas[name] = delta
    out = os.path.join(a.into, "edited"); os.makedirs(out, exist_ok=True)
    for fn in os.listdir(out): os.remove(os.path.join(out, fn))
    base_now = {}
    for name, delta in deltas.items():
        pt = parent_tensor(name)
        save_file({"tensor": (pt + delta).contiguous()}, os.path.join(out, name + ".safetensors"))
    print(f"staged crossover of {a.cands} ({len(deltas)} tensors) in {out}")


def cmd_status(a):
    st = state()
    print(json.dumps({k: st[k] for k in ("gen", "champion", "lineage")}, indent=1))
    for h in st["history"]:
        print(f"gen {h['gen']}: champion acc {h['champion_acc']}, winner {h['winner']}")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init"); p.add_argument("--force", action="store_true")
    p = sp.add_parser("new-gen"); p.add_argument("--editors", required=True); p.add_argument("--budget", type=int, default=6)
    p.add_argument("--max-cands", type=int, default=2); p.add_argument("--sel-n", type=int, default=800)
    p.add_argument("--focus", default=None, help="JSON {editor: suggested starting point}")
    p.add_argument("--protocol", choices=["direct", "think"], default="direct")
    p.add_argument("--template", default="editor_task.md")
    p.add_argument("--trace-bank", default=None, help="jsonl of base-model thinking traces shared with editors (phase 2)")
    p.add_argument("--sel-cap", type=int, default=None, help="thinking budget of the selection runs (D65: 8192)")
    p.add_argument("--sel-from", type=int, default=None, help="D68: reuse this generation's selection questions")
    sp.add_parser("candidates")
    p = sp.add_parser("select"); p.add_argument("--wait", action="store_true")
    p.add_argument("--priority", type=int, default=0)  # e.g. 9: run after editor passes
    p.add_argument("--parent-chunks", type=int, default=0, help="only pre-run the parent's side, in jobs of this size")
    p = sp.add_parser("decide"); p.add_argument("--z", type=float, default=2.0); p.add_argument("--force")
    p.add_argument("--bf-z", type=float, default=None, help="D65: also require both-finish paired z >= this (and > 0)")
    p.add_argument("--guard-z", type=float, default=None, help="D65: also require the transfer guard's paired z > this")
    p.add_argument("--keep", action="store_true")
    p = sp.add_parser("heldout"); p.add_argument("who", nargs="?", default="champion"); p.add_argument("--n", type=int, default=1000)
    p.add_argument("--set", choices=["mmlupro", "gpqa"], default="mmlupro")
    p.add_argument("--protocol", choices=["direct", "think"], default="direct")
    p.add_argument("--cap", type=int, help="thinking-mode token cap (default: gen.THINK_CAP; D60)")
    sp.add_parser("status")
    p = sp.add_parser("brief"); p.add_argument("editor", nargs="?")
    p = sp.add_parser("report"); p.add_argument("--gen", type=int)
    p = sp.add_parser("guard"); p.add_argument("--priority", type=int, default=9)
    p = sp.add_parser("crossover"); p.add_argument("cands"); p.add_argument("--into", required=True); p.add_argument("--gen", type=int)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "events.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cmd": a.cmd,
                            "args": {k: v for k, v in vars(a).items() if k != "cmd"}}) + "\n")
    {"init": cmd_init, "new-gen": cmd_new_gen, "candidates": cmd_candidates, "select": cmd_select,
     "decide": cmd_decide, "heldout": cmd_heldout, "status": cmd_status, "brief": cmd_brief, "report": cmd_report, "crossover": cmd_crossover,
     "guard": cmd_guard}[a.cmd](a)


if __name__ == "__main__":
    main()

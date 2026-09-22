"""GPU workloads, one subcommand each. Every one runs for --duration seconds.

    python -m gputel.workloads vision --mode train --model resnet18 --batch-size 64
    python -m gputel.workloads gradprobe --mode pgd
    python -m gputel.workloads llm --model Qwen/Qwen2.5-1.5B-Instruct
    python -m gputel.workloads idle | fft | nbody | mining | render

Run them through run_workloads.py, which wraps each one in a telemetry
recording. They are usable on their own for debugging.
"""
from __future__ import annotations

import argparse
import time


def _loop(duration, step, every=50, name="workload"):
    """Call step() until duration elapses, with periodic progress lines."""
    import torch
    t0, n = time.time(), 0
    while time.time() - t0 < duration:
        step()
        n += 1
        if n % every == 0:
            torch.cuda.synchronize()
            el = time.time() - t0
            print(f"  [{name}] {n} steps  {n / el:.2f}/s  {el:.0f}s", flush=True)
    torch.cuda.synchronize()
    print(f"[{name}] done: {n} steps in {time.time() - t0:.1f}s", flush=True)


# ── ML: training vs inference, matched ───────────────────────────────────────

def vision(a):
    """Train or infer with an identical model, batch shape and data path.

    The only difference between modes is the backward pass and optimiser step,
    so any separation the classifier finds comes from training itself.
    Synthetic batches are copied host->device every step, so PCIe traffic looks
    like a real dataloader's.
    """
    import torch
    import torch.nn as nn
    import torchvision

    dev = torch.device("cuda")
    model = getattr(torchvision.models, a.model)(weights=None, num_classes=10).to(dev)
    x_host = torch.randn(a.batch_size, 3, a.img, a.img).pin_memory()
    y_host = torch.randint(0, 10, (a.batch_size,)).pin_memory()

    if a.mode == "train":
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        lossfn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

        def step():
            x, y = x_host.to(dev, non_blocking=True), y_host.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=a.amp):
                loss = lossfn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    else:
        model.eval()

        def step():
            x = x_host.to(dev, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", enabled=a.amp):
                model(x)

    _loop(a.duration, step, every=100, name=f"{a.mode} {a.model}")


def gradprobe(a):
    """Backward passes that never train anything.

    pgd      adversarial attack: gradients w.r.t. the input
    ig       integrated-gradients attribution
    featviz  feature visualisation: optimises the input image
    pinn     physics-informed residual: autograd for input derivatives

    The model is frozen; its weights are verified unchanged at exit so the run
    fails loudly rather than producing mislabelled data.
    """
    import torch
    import torch.nn as nn
    import torchvision

    dev = torch.device("cuda")
    if a.mode == "pinn":
        model = nn.Sequential(nn.Linear(3, 256), nn.Tanh(), nn.Linear(256, 256),
                              nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                              nn.Linear(256, 1)).to(dev)
    else:
        model = torchvision.models.resnet18(weights=None, num_classes=10).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = [p.detach().clone() for p in model.parameters()]
    lossfn = nn.CrossEntropyLoss()
    bs, img, k = a.batch_size, 224, a.steps

    def step():
        if a.mode == "pgd":
            x = torch.randn(bs, 3, img, img, device=dev)
            y = torch.randint(0, 10, (bs,), device=dev)
            delta = torch.zeros_like(x, requires_grad=True)
            for _ in range(k):
                g, = torch.autograd.grad(lossfn(model(x + delta), y), delta)
                delta = (delta + 0.01 * g.sign()).clamp(-0.03, 0.03).detach().requires_grad_(True)
        elif a.mode == "ig":
            x = torch.randn(bs, 3, img, img, device=dev)
            y = torch.randint(0, 10, (bs,), device=dev)
            total = torch.zeros_like(x)
            for i in range(k):
                xi = (((i + 0.5) / k) * x).requires_grad_(True)
                g, = torch.autograd.grad(model(xi).gather(1, y[:, None]).sum(), xi)
                total += g
        elif a.mode == "featviz":
            im = torch.randn(bs, 3, img, img, device=dev, requires_grad=True)
            opt = torch.optim.Adam([im], lr=0.05)
            for _ in range(k):
                opt.zero_grad(set_to_none=True)
                (-model(im)[:, 0].mean()).backward()
                opt.step()
        else:
            pts = torch.rand(bs * 512, 3, device=dev, requires_grad=True)
            grad_u, = torch.autograd.grad(model(pts).sum(), pts, create_graph=True)
            lap = sum(torch.autograd.grad(grad_u[:, d].sum(), pts, retain_graph=True)[0][:, d]
                      for d in range(3))
            (lap ** 2).mean().item()

    _loop(a.duration, step, name=f"gradprobe {a.mode}")
    if any(not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before)):
        raise SystemExit("ERROR: model weights changed -- this would be training")


# Fixed prompts, identical for every model so the input is not a variable.
PROMPTS = [
    "Explain in detail how a heat pump moves thermal energy against a gradient.",
    "Summarise the causes and consequences of the 1970s oil shocks.",
    "Describe the differences between TCP and UDP and when each is preferred.",
    "Write a short technical note on why floating point addition is not associative.",
]


def llm(a):
    """Steady-state HuggingFace generation with a fixed serving configuration.

    min_new_tokens pins generation length: without it a model that emits EOS
    early does less work per call, and the telemetry reflects EOS behaviour
    rather than the model.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True).to("cuda")
    model.eval()
    print(f"[llm] {a.model}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B "
          f"params, arch={model.config.model_type}", flush=True)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(a.batch_size)]
    enc = tok(prompts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=a.seq_len).to("cuda")
    gen = dict(do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id)
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=4, min_new_tokens=4, **gen)   # warm-up

    def step():
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=a.max_new_tokens,
                           min_new_tokens=a.max_new_tokens, **gen)

    _loop(a.duration, step, every=20, name="llm")


# ── non-ML GPU work ──────────────────────────────────────────────────────────

def idle(a):
    """Nothing on the GPU. The baseline for the `other` class."""
    time.sleep(a.duration)


def fft(a):
    """Repeated 2-D complex FFTs: signal-processing style, bandwidth-bound."""
    import torch
    x = torch.randn(a.size, a.size, dtype=torch.complex64, device="cuda")

    def step():
        nonlocal x
        x = torch.fft.ifft2(torch.fft.fft2(x))
        x = x / x.abs().max()

    _loop(a.duration, step, name=f"fft {a.size}")


def nbody(a):
    """All-pairs gravitational N-body integration: compute-bound, no autograd."""
    import torch
    n = a.particles
    pos = torch.randn(n, 3, device="cuda")
    vel = torch.zeros(n, 3, device="cuda")
    mass = torch.rand(n, 1, device="cuda") + 0.5

    def step():
        nonlocal pos, vel
        d = pos[None, :, :] - pos[:, None, :]
        inv = (d.pow(2).sum(-1) + 1e-2).rsqrt().pow(3)
        acc = (d * (inv * mass.T)[..., None]).sum(1)
        vel = vel + 1e-4 * acc
        pos = pos + 1e-4 * vel

    _loop(a.duration, step, name=f"nbody {n}")


def mining(a):
    """Ethash-like proxy: random reads from a large DAG mixed with integer hashing.

    Memory-latency-bound with little arithmetic and no host traffic -- the
    signature of proof-of-work mining.
    """
    import torch
    dag = torch.randint(0, 2 ** 31 - 1, (a.dag_mb * 2 ** 18,), dtype=torch.int32, device="cuda")
    n = dag.numel()
    nonce = torch.arange(a.lanes, dtype=torch.int64, device="cuda")

    def step():
        nonlocal nonce
        mix = nonce.clone()
        for _ in range(64):
            idx = (mix * 2654435761) % n
            mix = (mix * 0x01000193) ^ dag[idx].to(torch.int64)
            mix = mix & 0x7FFFFFFF
        nonce = nonce + a.lanes

    _loop(a.duration, step, name="mining")


def render(a):
    """Rendering proxy: per-pixel ray-sphere shading, one frame per step.

    Each frame is copied back to the host, like a renderer writing frames out.
    """
    import torch
    h, w = a.height, a.width
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device="cuda"),
                            torch.linspace(-1, 1, w, device="cuda"), indexing="ij")
    rays = torch.stack([xs, ys, torch.ones_like(xs)], -1)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    centres = torch.randn(64, 3, device="cuda") + torch.tensor([0, 0, 6.0], device="cuda")
    frame = 0

    def step():
        nonlocal frame
        c = centres + 0.1 * torch.sin(torch.tensor(frame / 10.0, device="cuda"))
        b = rays @ c.T                                   # (h, w, spheres)
        disc = b ** 2 - (c.pow(2).sum(-1) - 1.0)
        t = torch.where(disc > 0, b - disc.clamp(min=0).sqrt(), torch.full_like(b, 1e9))
        tmin, idx = t.min(-1)
        hit = rays * tmin[..., None]
        normal = hit - c[idx]
        normal = normal / normal.norm(dim=-1, keepdim=True)
        shade = (normal @ torch.tensor([0.5, -0.5, -0.7], device="cuda")).clamp(min=0)
        img = torch.where(tmin < 1e8, shade, torch.zeros_like(shade))
        img.to("cpu", non_blocking=True)
        frame += 1

    _loop(a.duration, step, every=200, name="render")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gputel.workloads", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    def add(name, fn, **defaults):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        p.add_argument("--duration", type=int, default=600, help="seconds")
        p.set_defaults(fn=fn)
        return p

    p = add("vision", vision)
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--model", default="resnet18", help="torchvision model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--amp", action="store_true", help="mixed precision")

    p = add("gradprobe", gradprobe)
    p.add_argument("--mode", choices=["pgd", "ig", "featviz", "pinn"], required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=8, help="inner gradient steps per batch")

    p = add("llm", llm)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256, help="prompt padding length")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    add("idle", idle)
    add("fft", fft).add_argument("--size", type=int, default=4096)
    add("nbody", nbody).add_argument("--particles", type=int, default=16384)
    p = add("mining", mining)
    p.add_argument("--dag-mb", type=int, default=4096)
    p.add_argument("--lanes", type=int, default=1 << 20)
    p = add("render", render)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

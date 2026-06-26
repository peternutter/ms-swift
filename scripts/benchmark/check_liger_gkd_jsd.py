#!/usr/bin/env python3
"""Check Liger fused-linear GKD JSD against a dense reference.

This is intended as a local CUDA smoke test for SWIFT GKD. It verifies beta=0
(forward KL), beta=1 (reverse KL), and mixed JSD, then reports dense vs fused
forward+backward runtime and peak CUDA allocation.
"""

import argparse
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def dense_gkd_jsd(student_input, student_weight, teacher_input, teacher_weight, labels, beta, temperature, ignore_index):
    student_logits = (student_input @ student_weight.t()).float() / temperature
    with torch.no_grad():
        teacher_logits = (teacher_input @ teacher_weight.t()).float() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        per_vocab = torch.exp(teacher_log_probs) * (teacher_log_probs - student_log_probs)
    elif beta == 1:
        per_vocab = torch.exp(student_log_probs) * (student_log_probs - teacher_log_probs)
    else:
        log_beta = torch.log(torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device))
        log_1_minus_beta = torch.log1p(
            -torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device))
        mean_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + log_1_minus_beta, teacher_log_probs + log_beta]), dim=0)
        teacher_kl = torch.exp(teacher_log_probs) * (teacher_log_probs - mean_log_probs)
        student_kl = torch.exp(student_log_probs) * (student_log_probs - mean_log_probs)
        per_vocab = beta * teacher_kl + (1 - beta) * student_kl

    per_token = per_vocab.sum(dim=-1)
    mask = labels != ignore_index
    return per_token.masked_select(mask).sum() / mask.sum().clamp_min(1)


def clone_inputs(tokens, hidden, vocab, dtype, device, ignore_index):
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    student_input = torch.randn(tokens, hidden, generator=gen, device=device, dtype=dtype)
    teacher_input = torch.randn(tokens, hidden, generator=gen, device=device, dtype=dtype)
    student_weight = torch.randn(vocab, hidden, generator=gen, device=device, dtype=dtype) / hidden**0.5
    teacher_weight = torch.randn(vocab, hidden, generator=gen, device=device, dtype=dtype) / hidden**0.5
    labels = torch.randint(0, vocab, (tokens,), generator=gen, device=device)
    labels[::7] = ignore_index
    return student_input, student_weight, teacher_input, teacher_weight, labels


def run_once(loss_fn, inputs):
    student_input, student_weight, teacher_input, teacher_weight, labels = inputs
    student_input = student_input.detach().clone().requires_grad_(True)
    student_weight = student_weight.detach().clone().requires_grad_(True)
    teacher_input = teacher_input.detach().clone()
    teacher_weight = teacher_weight.detach().clone()
    loss = loss_fn(student_input, student_weight, teacher_input, teacher_weight, labels)
    loss.backward()
    return loss.detach(), student_input.grad.detach(), student_weight.grad.detach()


def bench(fn, inputs, iters, warmup, device):
    for _ in range(warmup):
        run_once(fn, inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        run_once(fn, inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_gb = float('nan')
    return (time.perf_counter() - start) / iters, peak_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--vocab', type=int, default=8192)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()

    if torch is None:
        print('SKIP: torch is not installed.')
        return 0

    try:
        from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    except ImportError as exc:
        print(f'SKIP: liger_kernel is not installed: {exc}')
        return 0

    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.bfloat16
    elif args.allow_cpu:
        device = torch.device('cpu')
        dtype = torch.float32
    else:
        print('SKIP: CUDA is not available. Use --allow-cpu for correctness-only CPU checking.')
        return 0

    ignore_index = -100
    base_inputs = clone_inputs(args.tokens, args.hidden, args.vocab, dtype, device, ignore_index)
    for beta in (0.0, 0.5, 1.0):
        fused_loss = LigerFusedLinearJSDLoss(
            weight_hard_loss=0.0,
            weight_soft_loss=1.0,
            beta=beta,
            ignore_index=ignore_index,
            temperature=args.temperature,
            compiled=False,
            chunk_size=args.chunk_size,
        )

        dense_fn = lambda s_i, s_w, t_i, t_w, labels: dense_gkd_jsd(
            s_i, s_w, t_i, t_w, labels, beta, args.temperature, ignore_index)
        fused_fn = lambda s_i, s_w, t_i, t_w, labels: fused_loss(
            student_input=s_i,
            student_weight=s_w,
            teacher_input=t_i,
            teacher_weight=t_w,
            true_labels=labels,
        )

        dense_loss, dense_grad_input, dense_grad_weight = run_once(dense_fn, base_inputs)
        fused_loss_value, fused_grad_input, fused_grad_weight = run_once(fused_fn, base_inputs)
        torch.testing.assert_close(fused_loss_value, dense_loss, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(fused_grad_input, dense_grad_input, rtol=5e-2, atol=5e-3)
        torch.testing.assert_close(fused_grad_weight, dense_grad_weight, rtol=5e-2, atol=5e-3)

        dense_time, dense_peak = bench(dense_fn, base_inputs, args.iters, args.warmup, device)
        fused_time, fused_peak = bench(fused_fn, base_inputs, args.iters, args.warmup, device)
        mem = '' if device.type != 'cuda' else f', peak_gb dense={dense_peak:.3f} fused={fused_peak:.3f}'
        print(
            f'beta={beta:g}: loss={fused_loss_value.item():.6f}, '
            f'time_ms dense={dense_time * 1000:.2f} fused={fused_time * 1000:.2f}{mem}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

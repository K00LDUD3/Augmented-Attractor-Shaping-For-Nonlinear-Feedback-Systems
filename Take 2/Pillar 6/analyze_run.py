import json, numpy as np

data = json.load(open(r'D:\Repos\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\Take 2\Pillar 6\6c_gali_both_batch\2026-04-13_04-09-26_2ea6cbee0a\logs\eval_history.json'))
efforts = [e['eval_total_effort'] for e in data]
errors = [e['eval_final_error'] for e in data]
itwae = [e['eval_itwae'] for e in data]

print('=== Run 3 Eval Summary ===')
print(f'Effort:  mean={np.mean(efforts):.1f}  min={np.min(efforts):.1f}  max={np.max(efforts):.1f}')
print(f'Error:   mean={np.mean(errors):.4f}  min={np.min(errors):.4f}')
print(f'ITWAE:   mean={np.mean(itwae):.2f}  min={np.min(itwae):.2f}')
print(f'PID baseline (single IC): Effort=4544.74  Error=0.0016  ITWAE=8.61')
print(f'Effort ratio vs PID: {np.mean(efforts)/4544.74:.2f}x')

mid = len(data) // 2
print(f'\nFirst half vs Second half:')
print(f'  Effort: {np.mean(efforts[:mid]):.1f} -> {np.mean(efforts[mid:]):.1f}')
print(f'  Error:  {np.mean(errors[:mid]):.4f} -> {np.mean(errors[mid:]):.4f}')
print(f'  ITWAE:  {np.mean(itwae[:mid]):.2f} -> {np.mean(itwae[mid:]):.2f}')

print(f'\nLast 10 evals:')
for e in data[-10:]:
    ep = e["episode"]
    eff = e["eval_total_effort"]
    err = e["eval_final_error"]
    itw = e["eval_itwae"]
    print(f'  Ep {ep:4d}  Eff={eff:8.1f}  Err={err:.4f}  ITWAE={itw:.2f}')

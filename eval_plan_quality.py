"""semantic_plan 规划层题目质量评估 (离线、纯图，不依赖 LLM)。"""
import json, collections, statistics
from autoqag.ops import load_all_stages; load_all_stages()
from autoqag.ops.base import PipelineContext
from autoqag.registry import STAGES

WD = '/tmp/spt'
ctx = PipelineContext(work_dir=WD, global_params={'domain': 'metamaterials'})
STAGES.get('semantic_plan')(per_type=12).run(ctx)

nodes = {json.loads(l)['node_id']: json.loads(l) for l in open(f'{WD}/nodes.jsonl', encoding='utf-8')}
plans = [json.loads(l) for l in open(f'{WD}/question_plans.jsonl', encoding='utf-8')]

def content(nid):
    d = nodes.get(nid, {})
    return (d.get('normalized_content') or d.get('content') or '')[:60]
def ntype(nid):
    return nodes.get(nid, {}).get('node_type', '?')
def grounded(nid):
    a = nodes.get(nid, {}).get('address', {})
    return bool(a.get('chunk_id') or a.get('span') or a.get('section_path'))

MIN = {  # 最小角色集 (与 roles.py 一致)
 'atomic':['concept'], 'numerical':['attribute','value','evidence'],
 'condition':['claim','condition_boundary','attribute_or_result','evidence_span'],
 'comparative':['object_A','object_B','shared_attribute','value_A','value_B','evidence_A','evidence_B'],
 'mechanism':['method_or_intervention','intermediate_mechanism','target_attribute','observed_result','supporting_evidence'],
 'table':['figure_or_table','evidence'], 'formula':['equation','evidence'],
 'multi_hop':['concept','intermediate_node','evidence','conclusion'],
 'cross_paper':['canonical_concept','paper_A_instance','paper_B_instance','aligned_attribute','evidence_A','evidence_B'],
 'summary':['section','claim'],
}
ABSTRACT = {'aligned_condition','comparison_criterion','cross_paper_relation','invalid_generalization_risk'}

N = len(plans)
print(f'=== 总览：{N} 个规划，覆盖 {len(set(p["question_type"] for p in plans))} 题型 ===')
print('题型分布:', dict(collections.Counter(p['question_type'] for p in plans)))
print('难度分布:', dict(collections.Counter(p['difficulty'] for p in plans)))

# ---- 质量维度 ----
role_ok = ev_ok = phys_ok = noshort = chunk_ok = paper_ok = constr_ok = 0
utils = []
for p in plans:
    qt = p['question_type']; ra = p['role_assignment']; nids = p['required_nodes']
    sb = p.get('score_breakdown', {}); utils.append(p.get('utility_score', 0))
    # 1. 角色完整：最小角色全部填满
    req = MIN.get(qt, [])
    if all(r in ABSTRACT or ra.get(r) for r in req): role_ok += 1
    # 2. 证据接地：所有 evidence 角色指向带物理地址的节点
    evr = [r for r in req if 'evidence' in r]
    if evr and all(grounded(ra.get(r)) for r in evr if ra.get(r)): ev_ok += 1
    elif not evr: ev_ok += 1
    # 3. 全部 required_nodes 是物理节点 (可追溯)
    if nids and all(n in nodes for n in nids): phys_ok += 1
    # 4. 无捷径 (shortcut 罚分=0)
    if sb.get('shortcut', 1) == 0: noshort += 1
    # 5. L3/L4 真跨 chunk
    chunks = {nodes.get(n,{}).get('address',{}).get('chunk_id','') for n in nids}; chunks.discard('')
    if p['difficulty'] in ('L3','L4'):
        if len(chunks) >= 2: chunk_ok += 1
    else: chunk_ok += 1
    # 6. cross_paper 真跨论文
    if qt == 'cross_paper':
        if len(set(p['paper_id_list'])) >= 2: paper_ok += 1
    else: paper_ok += 1
    # 7. 约束覆盖 (数值/公式/比较/图表题须 >=0.5)
    if qt in ('numerical','formula','comparative','table'):
        if sb.get('constraint',0) >= 0.5: constr_ok += 1
    else: constr_ok += 1

def pct(x): return f'{x}/{N} ({100*x/N:.0f}%)'
print('\n=== 质量维度 (规划层硬约束达成率) ===')
print('角色完整性 (最小角色集全填):', pct(role_ok))
print('证据物理接地 (evidence→可追溯节点):', pct(ev_ok))
print('全节点可追溯 (∈G0物理图):', pct(phys_ok))
print('抗捷径 (shortcut罚=0):', pct(noshort))
print('多跳真实性 (L3/L4真跨≥2chunk):', pct(chunk_ok))
print('跨文献真实性 (cross_paper跨≥2论文):', pct(paper_ok))
print('约束覆盖 (数值类题≥0.5):', pct(constr_ok))
print(f'\n综合效用分 utility: 均值={statistics.mean(utils):.2f} 中位={statistics.median(utils):.2f} 最小={min(utils):.2f} 最大={max(utils):.2f}')

# 子图规模 / 跨度
sizes = [len(p['required_nodes']) for p in plans]
spans = []
for p in plans:
    ch = {nodes.get(n,{}).get('address',{}).get('chunk_id','') for n in p['required_nodes']}; ch.discard('')
    spans.append(len(ch))
print(f'子图节点数: 均值={statistics.mean(sizes):.1f} 范围={min(sizes)}-{max(sizes)}')
print(f'跨 chunk 数: 均值={statistics.mean(spans):.1f} 范围={min(spans)}-{max(spans)}')

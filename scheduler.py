"""Scheduling core adapted from the executed home-care notebook."""


from dataclasses import dataclass, replace
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

TRAVEL_CENTS = 25
OVERTIME_CENTS = 120


def minute(hhmm):
    hour, mins = map(int, hhmm.split(":"))
    return hour * 60 + mins

def clock(value):
    return f"{value // 60:02d}:{value % 60:02d}"

@dataclass(frozen=True)
class Job:
    id: str
    site: str
    grade: int
    duration: int
    opens: int
    preferred_end: int
    deadline: int

@dataclass(frozen=True)
class Shift:
    id: str
    grade: int
    start: int
    regular_end: int
    hard_end: int
    shared: bool = False
    fee: int = 0
    origin: str = "D"
    destination: str = "D"

@dataclass(frozen=True)
class Scenario:
    name: str
    jobs: tuple
    shifts: tuple
    travel: dict
    budget: int

def job(number, grade, duration, opens, preferred_end, deadline):
    key = f"J{number:02d}"
    return Job(key, key, grade, duration, *map(minute, (opens, preferred_end, deadline)))

def shift(name, grade, start, regular_end, hard_end, fee=0, *, shared=False):
    return Shift(name, grade, *map(minute, (start, regular_end, hard_end)), shared, fee)

def make_travel(coords):
    # 示意路网，坐标为匿名网格；同点 0，跨点按 Manhattan 距离计算分钟。
    return {(a, b): (0 if a == b else 2 + 2 * (abs(x-u) + abs(y-v)))
            for a, (x, y) in coords.items() for b, (u, v) in coords.items()}

coordinates = dict(zip(
    ["D", *[f"J{i:02d}" for i in range(1, 17)]],
    [(4, 4), (2, 5), (5, 1), (6, 6), (7, 4), (1, 3), (2, 1), (4, 7),
     (1, 6), (6, 1), (8, 2), (4, 5), (8, 5), (3, 6), (3, 2), (7, 7), (3, 4)],
))
jobs = (
    job(1, 3, 35, "09:40", "09:45", "09:50"),
    job(2, 2, 25, "08:25", "08:55", "09:20"),
    job(3, 3, 35, "09:40", "09:45", "09:50"),
    job(4, 3, 40, "13:10", "13:30", "14:05"),
    job(5, 1, 25, "08:40", "09:10", "09:35"),
    job(6, 1, 25, "11:40", "12:05", "12:25"),
    job(7, 2, 30, "10:20", "10:45", "11:05"),
    job(8, 2, 30, "10:15", "10:30", "11:05"),
    job(9, 2, 35, "14:40", "15:05", "15:30"),
    job(10, 2, 35, "15:25", "15:45", "16:15"),
    job(11, 3, 35, "09:40", "09:45", "09:50"),
    job(12, 2, 35, "15:40", "15:55", "16:20"),
    job(13, 2, 30, "08:45", "09:10", "09:35"),
    job(14, 2, 30, "12:10", "12:30", "12:55"),
    job(15, 3, 35, "11:00", "11:25", "11:50"),
    job(16, 1, 25, "10:25", "10:45", "11:15"),
)
shifts = (
    shift("F01", 2, "08:00", "16:00", "16:40"),
    shift("F02", 3, "08:00", "16:00", "16:40"),
    shift("F03", 1, "08:00", "16:00", "16:40"),
    shift("F04", 3, "08:00", "16:00", "16:40"),
    shift("S01", 3, "09:15", "11:15", "11:15", 5500, shared=True),
    shift("S02", 2, "08:45", "11:00", "11:00", 4200, shared=True),
    shift("S03", 2, "14:00", "16:40", "16:40", 3800, shared=True),
    shift("S04", 3, "12:30", "16:40", "16:40", 6800, shared=True),
)
main = Scenario("共享补缺", jobs, shifts, make_travel(coordinates), 8000)

def update_window(scenario, job_id, opens, preferred_end, deadline, name):
    assert job_id in {j.id for j in scenario.jobs}
    windows = dict(zip(("opens", "preferred_end", "deadline"), map(minute, (opens, preferred_end, deadline))))
    return replace(scenario, name=name,
                   jobs=tuple(replace(j, **windows) if j.id == job_id else j for j in scenario.jobs))

enough = update_window(main, "J11", "12:45", "13:05", "13:20", "全职充足")
shortage = update_window(main, "J15", "09:40", "09:45", "09:50", "共享后仍短缺")
scenarios = [enough, main, shortage]

def fulltime(scenario):
    return replace(scenario, shifts=tuple(s for s in scenario.shifts if not s.shared), budget=0)

def validate_input(scenario):
    jobs, shifts, travel = scenario.jobs, scenario.shifts, scenario.travel
    if len({j.id for j in jobs}) != len(jobs) or len({s.id for s in shifts}) != len(shifts):
        raise ValueError("订单或人员编号重复")
    if type(scenario.budget) is not int or scenario.budget < 0:
        raise ValueError("预算应为非负整数分")
    for j in jobs:
        if any(type(x) is not int for x in (j.grade, j.duration, j.opens, j.preferred_end, j.deadline)):
            raise ValueError(f"{j.id} 存在非整数输入")
        if not (0 <= j.opens <= j.preferred_end <= j.deadline < 1440 and j.duration > 0 and j.grade > 0):
            raise ValueError(f"{j.id} 的开工窗、时长或资质无效")
    for s in shifts:
        if any(type(x) is not int for x in (s.grade, s.start, s.regular_end, s.hard_end, s.fee)):
            raise ValueError(f"{s.id} 存在非整数输入")
        if not (0 <= s.start < s.regular_end <= s.hard_end < 1440 and s.grade > 0 and s.fee >= 0):
            raise ValueError(f"{s.id} 的班次、资质或费用无效")
        if s.shared and s.regular_end != s.hard_end:
            raise ValueError(f"{s.id} 为共享班次，首版不允许加班")
        if not s.shared and s.fee:
            raise ValueError(f"{s.id} 全职固定工资不应计为借调费用")
    sites = {j.site for j in jobs} | {s.origin for s in shifts} | {s.destination for s in shifts}
    for a, b in product(sites, repeat=2):
        t = travel.get((a, b))
        if type(t) is not int or t < 0 or (a == b and t != 0):
            raise ValueError(f"缺少或无效的旅行边 {a} → {b}")
    # 保证直接旅行是可用下界，便于安全剪枝与解释单笔可达性。
    for a, b, c in product(sites, repeat=3):
        if travel[a, c] > travel[a, b] + travel[b, c]:
            raise ValueError(f"旅行分钟不满足最短路性质 {a} → {b} → {c}")


@dataclass(frozen=True)
class Visit:
    shift: str
    job: str
    seq: int
    arrival: int
    start: int
    end: int

@dataclass(frozen=True)
class Plan:
    visits: tuple
    unassigned: tuple

@dataclass(frozen=True)
class Metrics:
    covered: int
    on_time: int
    late: int
    travel: int
    waiting: int
    overtime: int
    shared_cents: int
    cost: int
    shared_ids: tuple

def check_plan(scenario, plan):
    # 检查器独立读取明细，不使用贪心解码器或 CP-SAT 变量。
    jobs = {j.id: j for j in scenario.jobs}
    shifts = {s.id: s for s in scenario.shifts}
    assigned = [v.job for v in plan.visits]
    if len(assigned) != len(set(assigned)):
        raise ValueError("重复分配")
    if len(set(plan.unassigned)) != len(plan.unassigned):
        raise ValueError("未排单清单重复")
    if set(assigned) & set(plan.unassigned) or set(assigned) | set(plan.unassigned) != set(jobs):
        raise ValueError("明细和未排单清单必须完整分割输入订单")
    if any(v.shift not in shifts for v in plan.visits):
        raise ValueError("存在未知或已撤销的班次")

    on_time = late = travel = waiting = overtime = fee = 0
    shared_ids = []
    for staff in shifts.values():
        visits = sorted((v for v in plan.visits if v.shift == staff.id), key=lambda v: v.seq)
        if not visits:
            continue
        previous_end, previous_site = staff.start, staff.origin
        for seq, v in enumerate(visits, 1):
            j = jobs[v.job]
            if v.seq != seq:
                raise ValueError(f"{staff.id}/{v.job} 访问序号无效")
            if staff.grade < j.grade:
                raise ValueError(f"{staff.id}/{v.job} 资质不足")
            leg = scenario.travel[previous_site, j.site]
            if v.arrival != previous_end + leg or v.start < v.arrival:
                raise ValueError(f"{staff.id}/{v.job} 路程时间冲突")
            if not j.opens <= v.start <= j.deadline:
                raise ValueError(f"{staff.id}/{v.job} 开工超硬窗")
            if v.end != v.start + j.duration:
                raise ValueError(f"{staff.id}/{v.job} 服务时长不符")
            on_time += v.start <= j.preferred_end
            late += max(v.start - j.preferred_end, 0)
            travel += leg
            waiting += v.start - v.arrival
            previous_end, previous_site = v.end, j.site

        return_leg = scenario.travel[previous_site, staff.destination]
        returned = previous_end + return_leg
        if returned > staff.hard_end:
            raise ValueError(f"{staff.id}/{visits[-1].job} 返程超班")
        travel += return_leg
        overtime += max(returned - staff.regular_end, 0)
        if staff.shared:
            fee += staff.fee
            shared_ids.append(staff.id)

    if fee > scenario.budget:
        raise ValueError("超出借调预算")
    cost = TRAVEL_CENTS * travel + OVERTIME_CENTS * overtime + fee
    return Metrics(len(assigned), on_time, late, travel, waiting, overtime, fee, cost, tuple(shared_ids))

def score(metrics):
    # 字典序优先最大覆盖，其次最小迟到分钟，最后最小增量费用。
    return (-metrics.covered, metrics.late, metrics.cost)

def individually_possible(job, staff, travel):
    first = max(job.opens, staff.start + travel[staff.origin, job.site])
    return (staff.grade >= job.grade and first <= job.deadline
            and first + job.duration + travel[job.site, staff.destination] <= staff.hard_end)

def explain_unassigned(scenario, job_id):
    j = next(j for j in scenario.jobs if j.id == job_id)
    skilled = [s for s in scenario.shifts if s.grade >= j.grade]
    if not skilled:
        return "当前可用班次中无合格人员"
    possible = [s for s in skilled if individually_possible(j, s, scenario.travel)]
    if not possible:
        return "合格班次单独派单也无法在硬窗内开工并按时返程"
    if all(s.fee > scenario.budget for s in possible):
        return "单独可排的共享班次均超过当前借调预算"
    return "当前方案未安排；单独派单可行"


@dataclass(frozen=True)
class Route:
    visits: tuple
    late: int
    cost: int

EMPTY = Route((), 0, 0)

def lay_out(scenario, staff, queue):
    if not queue:
        return EMPTY
    visits, now, site = [], staff.start, staff.origin
    late = drive = 0
    for seq, j in enumerate(queue, 1):
        leg = scenario.travel[site, j.site]
        arrival = now + leg
        start = max(arrival, j.opens)
        if start > j.deadline:
            return None
        now = start + j.duration
        site = j.site
        drive += leg
        late += max(start - j.preferred_end, 0)
        visits.append(Visit(staff.id, j.id, seq, arrival, start, now))
    back = scenario.travel[site, staff.destination]
    returned = now + back
    if returned > staff.hard_end:
        return None
    cost = (drive + back) * TRAVEL_CENTS + max(returned - staff.regular_end, 0) * OVERTIME_CENTS + staff.fee
    return Route(tuple(visits), late, cost)

def insertion_plan(scenario, order, preference=None):
    shifts = scenario.shifts
    queues, timelines = [[] for _ in shifts], [EMPTY for _ in shifts]
    unassigned = []
    used_fee = 0
    for i in order:
        j = scenario.jobs[i]
        best = None
        for k, staff in enumerate(shifts):
            queue = queues[k]
            activation = staff.fee if not queue else 0
            if staff.grade < j.grade or used_fee + activation > scenario.budget:
                continue
            if not individually_possible(j, staff, scenario.travel):
                continue
            staff_key = 0 if preference is None else float(preference[i, k])
            current = timelines[k]
            for slot in range(len(queue) + 1):
                candidate = queue[:slot] + [j] + queue[slot:]
                layout = lay_out(scenario, staff, candidate)
                if layout is None:
                    continue
                # 搜索解码时，岗位偏好优先级可引导尝试不同分配。实际评价仍用统一目标。
                key = (staff_key, layout.late-current.late, layout.cost-current.cost, k, slot)
                if best is None or key < best[0]:
                    best = key, k, candidate, layout, activation
        if best is None:
            unassigned.append(j.id)
        else:
            _, k, queues[k], timelines[k], activation = best
            used_fee += activation
    return Plan(tuple(v for route in timelines for v in route.visits), tuple(sorted(unassigned)))

def greedy(scenario):
    order = sorted(range(len(scenario.jobs)), key=lambda i: (
        scenario.jobs[i].deadline, -scenario.jobs[i].grade, scenario.jobs[i].id
    ))
    return insertion_plan(scenario, order)

def decode_keys(scenario, x):
    n, k = len(scenario.jobs), len(scenario.shifts)
    order = np.argsort(x[:n], kind="stable")
    return insertion_plan(scenario, order, x[n:].reshape(n, k))


@dataclass(frozen=True)
class ExactResult:
    plan: Plan
    status: str
    seconds: float
    objective: int
    bound: float | None

def solve_cp(scenario, seconds=5.0):
    started = perf_counter()
    model = cp_model.CpModel()
    jobs, shifts, travel = scenario.jobs, scenario.shifts, scenario.travel
    n = len(jobs)
    times = [model.new_int_var(j.opens, j.deadline, f"start_{j.id}") for j in jobs]
    served = [model.new_bool_var(f"served_{j.id}") for j in jobs]
    penalties = []
    for i, j in enumerate(jobs):
        model.add(times[i] == j.opens).only_enforce_if(served[i].Not())
        late = model.new_int_var(0, j.deadline - j.preferred_end, f"late_{j.id}")
        model.add_max_equality(late, [0, times[i] - j.preferred_end])
        penalties.append(late)

    assigned = [[] for _ in jobs]
    shift_vars, costs, shared_fees = [], [], []
    for s in shifts:
        active = model.new_bool_var(f"use_{s.id}")
        returned = model.new_int_var(s.start, s.hard_end, f"return_{s.id}")
        overtime = model.new_int_var(0, s.hard_end - s.regular_end, f"ot_{s.id}")
        model.add(returned == s.start).only_enforce_if(active.Not())
        model.add_max_equality(overtime, [0, returned - s.regular_end])
        costs += [overtime * OVERTIME_CENTS, active * s.fee]
        if s.shared:
            shared_fees.append(active * s.fee)

        candidates = [i for i, j in enumerate(jobs) if individually_possible(j, s, travel)]
        used_nodes = {i: model.new_bool_var(f"{s.id}_{jobs[i].id}") for i in candidates}
        for i in candidates:
            assigned[i].append(used_nodes[i])
        model.add(sum(used_nodes.values()) >= active)
        model.add(sum(used_nodes.values()) <= n * active)
        arcs = [(0, 0, active.Not())]
        edge_vars = {}
        for i in range(n):
            stay = used_nodes[i].Not() if i in used_nodes else model.new_constant(1)
            arcs.append((i + 1, i + 1, stay))

        def add_edge(a, b, minutes):
            edge = model.new_bool_var(f"{s.id}_{a}_to_{b}")
            arcs.append((a, b, edge))
            edge_vars[a, b] = edge
            costs.append(edge * minutes * TRAVEL_CENTS)
            return edge

        for i in candidates:
            j = jobs[i]
            outgoing = travel[s.origin, j.site]
            incoming = travel[j.site, s.destination]
            first = add_edge(0, i + 1, outgoing)
            last = add_edge(i + 1, 0, incoming)
            model.add(times[i] >= s.start + outgoing).only_enforce_if(first)
            model.add(returned == times[i] + j.duration + incoming).only_enforce_if(last)
            for q in candidates:
                if q == i:
                    continue
                other = jobs[q]
                leg = travel[j.site, other.site]
                if j.opens + j.duration + leg > other.deadline:
                    continue
                edge = add_edge(i + 1, q + 1, leg)
                model.add(times[q] >= times[i] + j.duration + leg).only_enforce_if(edge)
        model.add_circuit(arcs)
        shift_vars.append((s, active, used_nodes, edge_vars))

    for i in range(n):
        model.add(sum(assigned[i]) == served[i])
    model.add(sum(shared_fees) <= scenario.budget)

    # 已服务 n 个订单时，跨点行程最多 n + 活跃班次数。
    cost_upper = (n + len(shifts)) * max(travel.values()) * TRAVEL_CENTS + sum(
        (s.hard_end - s.regular_end) * OVERTIME_CENTS + s.fee for s in shifts
    )
    late_upper = sum(j.deadline - j.preferred_end for j in jobs)
    weight_late = cost_upper + 1
    weight_cover = (late_upper + 1) * weight_late
    objective = (n - sum(served)) * weight_cover + sum(penalties) * weight_late + sum(costs)
    model.minimize(objective)

    # 只使用同输入下的规则排程作提示，加快找到第一份可行解。
    warm = greedy(scenario)
    lookup = {v.job: v for v in warm.visits}
    id_index = {j.id: i for i, j in enumerate(jobs)}
    for i, j in enumerate(jobs):
        model.add_hint(served[i], int(j.id in lookup))
        model.add_hint(times[i], lookup[j.id].start if j.id in lookup else j.opens)
    for s, active, nodes, edges in shift_vars:
        route = [id_index[v.job] + 1 for v in warm.visits if v.shift == s.id]
        chosen = set(zip([0] + route, route + [0]))
        model.add_hint(active, int(bool(route)))
        for i, variable in nodes.items():
            model.add_hint(variable, int(i + 1 in route))
        for edge, variable in edges.items():
            model.add_hint(variable, int(edge in chosen))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 2026
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        # 无解返回状态和已核查的规则方案，避免将其当成 CP-SAT 解。
        if status != cp_model.UNKNOWN:
            raise RuntimeError(f"CP-SAT 异常状态 {status_name}")
        m = check_plan(scenario, warm)
        value = (n-m.covered)*weight_cover + m.late*weight_late + m.cost
        return ExactResult(warm, f"{status_name} / 回退规则", perf_counter()-started, value, None)

    visits = []
    for s, active, _, edges in shift_vars:
        if not solver.boolean_value(active):
            continue
        successors = {a: b for (a, b), var in edges.items() if solver.boolean_value(var)}
        current, previous_end, site, seq = successors[0], s.start, s.origin, 1
        while current:
            i = current - 1
            j = jobs[i]
            start = solver.value(times[i])
            visits.append(Visit(s.id, j.id, seq, previous_end + travel[site, j.site], start, start+j.duration))
            previous_end, site, seq = start + j.duration, j.site, seq + 1
            current = successors[current]

    plan = Plan(tuple(visits), tuple(j.id for i, j in enumerate(jobs) if not solver.boolean_value(served[i])))
    metrics = check_plan(scenario, plan)
    reconstructed = (n-metrics.covered)*weight_cover + metrics.late*weight_late + metrics.cost
    assert reconstructed == round(solver.objective_value)
    return ExactResult(plan, status_name, perf_counter()-started, reconstructed, float(solver.best_objective_bound))


@dataclass(frozen=True)
class SearchResult:
    plan: Plan
    metrics: Metrics
    trace: tuple
    seconds: float
    evaluations: int

def seed_keys(scenario):
    n, k = len(scenario.jobs), len(scenario.shifts)
    order = sorted(range(n), key=lambda i: (scenario.jobs[i].deadline, -scenario.jobs[i].grade, scenario.jobs[i].id))
    x = np.full(n * (1 + k), 0.5)
    x[order] = np.linspace(0.1, 0.9, n)
    return x

def whale_step(x, best, peer, decay, rng):
    # WOA 的围捕/探索和螺旋两种更新。
    a = decay * (2 * rng.random() - 1)
    c = 2 * rng.random()
    if rng.random() < 0.5:
        anchor = best if abs(a) < 1 else peer
        return anchor - a * np.abs(c * anchor - x)
    spiral = rng.uniform(-1, 1)
    return best + np.abs(best - x) * np.exp(spiral) * np.cos(2 * np.pi * spiral)

def swarm_search(scenario, algorithm, seed, population=18, iterations=32):
    if algorithm not in {"WOA", "PSO", "WOA-PSO"}:
        raise ValueError(f"未知算法 {algorithm}")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    dim = len(scenario.jobs) * (1 + len(scenario.shifts))
    points = rng.random((population, dim))
    points[0] = seed_keys(scenario)
    velocity = rng.uniform(-0.1, 0.1, points.shape)

    def evaluate(x):
        plan = decode_keys(scenario, x)
        return plan, check_plan(scenario, plan)

    records = [evaluate(x) for x in points]
    personal_x = points.copy()
    personal_records = records.copy()
    trace = []
    for step in range(iterations + 1):
        lead = min(range(population), key=lambda p: score(personal_records[p][1]))
        global_x = personal_x[lead].copy()
        best_plan, best_metrics = personal_records[lead]
        trace.append(best_metrics)
        if step == iterations:
            break
        decay = 2 * (1 - step / max(iterations, 1))
        proposals = np.empty_like(points)
        for p, x in enumerate(points):
            if algorithm != "WOA":
                personal_push = 1.45 * rng.random(dim) * (personal_x[p] - x)
                group_push = 1.45 * rng.random(dim) * (global_x - x)
                velocity[p] = np.clip(0.65 * velocity[p] + personal_push + group_push, -0.28, 0.28)
            if algorithm == "PSO":
                proposals[p] = x + velocity[p]
                continue
            if algorithm == "WOA-PSO":
                # 三个候选历史最好解中选一个领跑者，替代纯 WOA 的随机探索伙伴。
                tournament = rng.choice(population, size=3, replace=False)
                winner = min(tournament, key=lambda q: score(personal_records[q][1]))
                peer = personal_x[winner]
            else:
                peer = points[rng.integers(population)]
            whale = whale_step(x, global_x, peer, decay, rng)
            proposals[p] = whale if algorithm == "WOA" else 0.65 * whale + 0.35 * (x + velocity[p])
        # 极少量随机重置仅用于混合版，缓解位置向边界聚集。消融比较的是整套组合。
        if algorithm == "WOA-PSO":
            jitter = rng.random(proposals.shape) < 0.025
            proposals[jitter] = rng.random(int(jitter.sum()))
        points = np.clip(proposals, 0, 1)
        for p, x in enumerate(points):
            candidate = evaluate(x)
            if score(candidate[1]) < score(personal_records[p][1]):
                personal_records[p] = candidate
                personal_x[p] = x
    return SearchResult(best_plan, best_metrics, tuple(trace), perf_counter()-started, population * (iterations+1))


def order_detail(scenario, plan, version):
    shifts = {s.id: s for s in scenario.shifts}
    visits = {v.job: v for v in plan.visits}
    rows = []
    for job in scenario.jobs:
        visit = visits.get(job.id)
        rows.append({
            "版本": version, "订单": job.id, "资质需求": job.grade,
            "期望开工": f"{clock(job.opens)}–{clock(job.preferred_end)}", "最迟开工": clock(job.deadline),
            "人员": visit.shift if visit else "",
            "人员等级": shifts[visit.shift].grade if visit else None, "访问序号": visit.seq if visit else None,
            "计划到达": clock(visit.arrival) if visit else "", "计划开工": clock(visit.start) if visit else "",
            "计划结束": clock(visit.end) if visit else "", "是否准时": (visit.start <= job.preferred_end) if visit else None,
            "等待分钟": visit.start - visit.arrival if visit else None,
            "未安排说明": explain_unassigned(scenario, job.id) if not visit else "",
        })
    return pd.DataFrame(rows).sort_values(["访问序号", "订单"], na_position="last").sort_values(
        "人员", kind="stable", key=lambda column: column.replace("", "ZZZ")
    ).reset_index(drop=True)

def staff_bill(scenario, plan):
    jobs = {j.id: j for j in scenario.jobs}
    result = []
    for s in scenario.shifts:
        visits = sorted((v for v in plan.visits if v.shift == s.id), key=lambda v: v.seq)
        now, site, drive, wait, work = s.start, s.origin, 0, 0, 0
        for v in visits:
            j = jobs[v.job]
            drive += scenario.travel[site, j.site]
            wait += v.start - v.arrival
            work += j.duration
            now, site = v.end, j.site
        back = scenario.travel[site, s.destination] if visits else 0
        drive += back
        ot = max(now+back-s.regular_end, 0)
        fee = s.fee if visits else 0
        cents = drive * TRAVEL_CENTS + ot * OVERTIME_CENTS + fee
        result.append({"班次": s.id, "订单数": len(visits), "路线": " → ".join(
                           [s.origin, *[v.job for v in visits], s.destination]) if visits else "未启用",
                       "返程时刻": clock(now+back) if visits else "", "服务分钟": work, "旅行分钟": drive,
                       "等待分钟": wait, "加班分钟": ot, "交通费用（元）": drive*TRAVEL_CENTS/100,
                       "加班费用（元）": ot*OVERTIME_CENTS/100, "共享费用（元）": fee/100,
                       "增量费用（元）": cents/100})
    table = pd.DataFrame(result)
    m = check_plan(scenario, plan)
    assert round(table["增量费用（元）"].sum()*100) == m.cost
    assert table["旅行分钟"].sum() == m.travel and table["加班分钟"].sum() == m.overtime
    return table

"""Interactive scenario comparison and versioned worklist export."""
from dataclasses import replace
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from desk_model import CheckedPlan, Settings
from scheduler import (
    check_plan, clock, fulltime, greedy, minute, order_detail,
    scenarios, solve_cp, staff_bill, swarm_search, validate_input,
)

SCENARIOS = {scenario.name: scenario for scenario in scenarios}
SCENE_NOTES = {
    "全职充足": "三级预约错开，检验全职能否独立覆盖全部订单。",
    "共享补缺": "三笔三级预约集中在 09:40–09:50，核查共享班次能否补上缺口。",
    "共享后仍短缺": "四笔三级预约集中在同一时段，检验开放共享后的剩余缺口。",
}
SHARED = {s.id: s for s in SCENARIOS["共享补缺"].shifts if s.shared}
PLAN_NAMES = {"A": "规则全职", "B": "全职优化", "C": "共享优化"}
PLAN_NOTES = {"A": "按预约截止时间插单", "B": "CP-SAT · 仅全职", "C": "CP-SAT · 开放候选班次"}


@st.cache_data(show_spinner=False)
def solve(settings: Settings) -> dict[str, CheckedPlan]:
    original = SCENARIOS[settings.scene]
    internal = fulltime(original)
    shared = replace(original, budget=settings.budget * 100, shifts=tuple(
        s for s in original.shifts if not s.shared or s.id in settings.candidates
    ))
    validate_input(internal)
    validate_input(shared)
    rule = greedy(internal)
    b, c = solve_cp(internal), solve_cp(shared)
    return {key: CheckedPlan(case, plan, check_plan(case, plan), status)
            for key, case, plan, status in (
                ("A", internal, rule, "规则可行"),
                ("B", internal, b.plan, b.status),
                ("C", shared, c.plan, c.status),
            )}


@st.cache_data(show_spinner=False)
def algorithm_table(settings: Settings) -> pd.DataFrame:
    scenario = solve(settings)["C"].scenario
    rows = []
    for name in ("WOA", "PSO", "WOA-PSO"):
        for seed in (11, 23, 37):
            result = swarm_search(scenario, name, seed)
            m = result.metrics
            rows.append({"算法": name, "种子": seed, "覆盖单": m.covered, "迟到分钟": m.late,
                         "增量费用（元）": m.cost / 100, "评价次数": result.evaluations,
                         "耗时（秒）": round(result.seconds, 3)})
    return pd.DataFrame(rows)


def gantt(checked: CheckedPlan) -> go.Figure:
    colors = {"旅行": "#9CAFC4", "空档": "#F3D98B", "服务": "#188477"}
    marks = {label: {"shift": [], "length": [], "start": [], "text": [], "hover": []}
             for label in colors}
    jobs = {job.id: job for job in checked.scenario.jobs}
    used = []

    def add(label, s, start, end, order=""):
        if start >= end:
            return
        data = marks[label]
        data["shift"].append(s.id)
        data["length"].append(end - start)
        data["start"].append(start)
        data["text"].append(order)
        data["hover"].append(f"{s.id} · {label} · {clock(start)}–{clock(end)}"
                             + (f" · {order}" if order else ""))

    for s in checked.scenario.shifts:
        visits = sorted((v for v in checked.plan.visits if v.shift == s.id), key=lambda v: v.seq)
        if not visits:
            continue
        used.append(s.id)
        previous_end = s.start
        for visit in visits:
            add("旅行", s, previous_end, visit.arrival)
            add("空档", s, visit.arrival, visit.start)
            add("服务", s, visit.start, visit.end, visit.job)
            previous_end = visit.end
        home = checked.scenario.travel[jobs[visits[-1].job].site, s.destination]
        add("旅行", s, previous_end, previous_end + home)

    chart = go.Figure()
    for label, d in marks.items():
        chart.add_trace(go.Bar(
            name=label, y=d["shift"], x=d["length"], base=d["start"], orientation="h",
            marker_color=colors[label], width=0.57, text=d["text"], textposition="inside",
            insidetextanchor="middle", textfont={"color": "white", "size": 12},
            customdata=d["hover"], hovertemplate="%{customdata}<extra></extra>",
        ))
    ticks = list(range(minute("08:00"), minute("17:00") + 1, 60))
    chart.update_layout(
        barmode="overlay", height=max(245, 66 + 43 * len(used)),
        margin={"l": 45, "r": 20, "t": 8, "b": 36}, paper_bgcolor="white", plot_bgcolor="white",
        font={"color": "#31485A", "size": 12}, hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": -0.25, "traceorder": "normal"},
        xaxis={"tickvals": ticks, "ticktext": [clock(x) for x in ticks], "range": [475, 1010],
               "gridcolor": "#E7EDF3", "zeroline": False, "fixedrange": True},
        yaxis={"categoryorder": "array", "categoryarray": used, "autorange": "reversed", "fixedrange": True},
    )
    return chart


def changed_orders(before: CheckedPlan, after: CheckedPlan) -> pd.DataFrame:
    old = {v.job: v for v in before.plan.visits}
    new = {v.job: v for v in after.plan.visits}
    rows = []
    for job in after.scenario.jobs:
        left, right = old.get(job.id), new.get(job.id)
        left_key = (left.shift, left.start) if left else None
        right_key = (right.shift, right.start) if right else None
        if left_key == right_key:
            continue
        if right is None:
            kind = "转为未安排"
        elif left is None:
            kind = "新增覆盖"
        elif left.shift != right.shift:
            kind = "更换人员"
        else:
            kind = "调整开工"
        rows.append({
            "订单": job.id, "变化": kind,
            "原方案": f"{left.shift} · {clock(left.start)}" if left else "未安排",
            "新方案": f"{right.shift} · {clock(right.start)}" if right else "未安排",
        })
    return pd.DataFrame(rows, columns=["订单", "变化", "原方案", "新方案"])


def cards(plans: dict[str, CheckedPlan], focused: str):
    for key, column in zip(("A", "B", "C"), st.columns(3)):
        checked = plans[key]
        m = checked.metrics
        share = ", ".join(m.shared_ids) or "无共享"
        style = "case-card active" if key == focused else "case-card"
        bar = 100 * m.covered / len(checked.scenario.jobs)
        column.html(f"""
        <div class="{style}">
          <div class="card-top"><span class="case-key">{key}</span><span class="card-status">{escape(checked.status)}</span></div>
          <div class="card-title">{PLAN_NAMES[key]}</div>
          <div class="card-note">{PLAN_NOTES[key]}</div>
          <div class="card-number">{m.covered}<span> / {len(checked.scenario.jobs)} 单</span></div>
          <div class="coverage"><span style="width:{bar:.1f}%"></span></div>
          <div class="card-foot"><span>准时 <b>{m.on_time} 单</b></span>
          <span>费用 <b>¥{m.cost/100:.2f}</b></span></div>
          <div class="card-note">启用班次 · {escape(share)}</div>
        </div>""")


def withdraw_s01():
    st.session_state.candidates = [item for item in st.session_state.candidates if item != "S01"]
    st.session_state.apply_withdrawal = True


st.set_page_config(page_title="家庭护理调度台", page_icon="🗓️", layout="wide")
st.html("""
<style>
.stApp {background:#F7F9FC;}
[data-testid="stMainBlockContainer"] {max-width:1280px;padding-top:2rem;padding-bottom:2.2rem;}
[data-testid="stSidebar"] {background:#EDF3F7;}
.hero {border-radius:20px;padding:26px 34px;background:linear-gradient(104deg,#16344C,#164655);color:white;}
.hero .eyebrow {font-size:11px;letter-spacing:.17em;color:#A9D8DA;font-weight:700;}
.hero h1 {font-size:clamp(26px,4vw,40px);line-height:1.2;letter-spacing:-.03em;padding:7px 0 5px;margin:0;color:white;}
.hero p {margin:0;color:#D5E5EC;font-size:15px;}
.hero .tags {display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;}
.hero .tags span {background:#FFFFFF16;border:1px solid #FFFFFF28;border-radius:999px;font-size:12px;padding:4px 10px;}
.case-card {background:white;border:1px solid #DFE8EE;border-radius:16px;padding:13px 16px;box-shadow:0 3px 9px #213B4B08;}
.case-card.active {border-color:#188477;box-shadow:0 0 0 1px #18847724;}
.card-top {display:flex;justify-content:space-between;align-items:center;}
.case-key {background:#E7F4F1;color:#12675E;font-weight:800;padding:4px 10px;border-radius:7px;font-size:12px;}
.card-status {font-size:11px;color:#406374;}
.card-title {font-size:17px;color:#132B3B;font-weight:800;margin-top:8px;}
.card-note {font-size:12px;color:#506877;line-height:1.5;margin-top:3px;}
.card-number {font-size:31px;font-weight:800;line-height:1.1;letter-spacing:-.04em;margin-top:8px;color:#15374B;}
.card-number span {font-size:13px;font-weight:600;color:#58717F;letter-spacing:0;}
.coverage {height:5px;border-radius:99px;background:#E7EDF0;margin:9px 0;}
.coverage span {display:block;height:5px;background:#188477;border-radius:99px;}
.card-foot {display:flex;gap:12px;justify-content:space-between;color:#506877;font-size:12px;}
.card-foot b {color:#143C4D;}
.delta {padding:11px 15px;background:#E9F4F1;border-radius:12px;border-left:4px solid #188477;color:#183D45;}
.delta b {color:#135B54;}
@media(max-width:700px){.hero{padding:20px 18px}.card-foot{flex-wrap:wrap;gap:2px;}}
</style>
<div class="hero">
  <div class="eyebrow">HOMECARE / DISPATCH DESK</div>
  <h1>家庭护理调度台</h1>
  <p>判断自有人手是否够用，核对借调班次能补上哪一单。</p>
  <div class="tags"><span>仿真案例</span><span>16 笔预约</span><span>4 名全职</span><span>4 个候选共享班次</span></div>
</div>
""")

state = st.session_state
if "applied" not in state:
    state.applied = Settings("共享补缺", 80, tuple(SHARED))
    state.version = 1
    state.previous = None
    state.confirmed = None

with st.sidebar:
    st.subheader("排程输入")
    scene = st.selectbox("订单情景", options=list(SCENARIOS), index=1, key="scene")
    st.caption(SCENE_NOTES[scene])
    budget = st.number_input("共享预算（元）", 0, 300, 80, step=1, key="budget")
    candidates = st.multiselect("可用共享班次", options=list(SHARED), default=list(SHARED),
                                key="candidates", placeholder="请选择可用班次")
    for s in SHARED.values():
        st.caption(f"{s.id} · {s.grade} 级 · {clock(s.start)}–{clock(s.hard_end)} · ¥{s.fee/100:.0f}")
    update = st.button("生成 / 更新三套方案", type="primary", key="generate", width="stretch")
    st.button("撤销 S01 并重排", on_click=withdraw_s01, disabled="S01" not in candidates,
              key="withdraw", width="stretch")
    st.caption("预算只约束共享班次费。交通和加班独立汇总。")

draft = Settings(scene, int(budget), tuple(s for s in SHARED if s in candidates))
if update or state.pop("apply_withdrawal", False):
    if draft != state.applied:
        state.previous = (state.version, state.applied)
        state.applied = draft
        state.version += 1
        state.confirmed = None

dirty = draft != state.applied
version = f"V{state.version:02}"
with st.spinner("正在生成并校验三套排程…"):
    plans = solve(state.applied)

main, choice = st.columns([2, 1], vertical_alignment="bottom")
with main:
    st.subheader("方案对照", anchor=False)
    st.caption(f"{version} · {state.applied.scene} · 共享预算 ¥{state.applied.budget} · 已逐项校验")
with choice:
    focused = st.radio("查看方案", ("A", "B", "C"), index=2, horizontal=True,
                       format_func=lambda x: f"{x} {PLAN_NAMES[x]}", key="focus")
if dirty:
    st.warning("输入已更改。生成新方案后才可确认及下载对应工作单。")
cards(plans, focused)

before, after = plans["B"], plans["C"]
m1, m2 = before.metrics, after.metrics
newly_covered = sorted(set(before.plan.unassigned) - set(after.plan.unassigned))
dropped = sorted(set(after.plan.unassigned) - set(before.plan.unassigned))
if m2.covered > m1.covered:
    signal = f"共享方案净增 <b>{m2.covered-m1.covered} 单</b>"
elif m2.covered < m1.covered:
    signal = "当前返回的共享方案覆盖更少，请结合求解状态核对"
else:
    signal = f"共享方案覆盖与全职相同 · {m2.covered} 单"
if newly_covered:
    signal += f" · 新增 {', '.join(newly_covered)}"
if dropped:
    signal += f" · 失去 {', '.join(dropped)}"
share_text = "、".join(m2.shared_ids) or "无"
st.html(f'<div class="delta"><b>B → C</b>　{escape(share_text)}　·　{signal}'
        f'　·　增量费用 <b>{(m2.cost-m1.cost)/100:+.2f} 元</b></div>')

if state.previous and state.previous[1].scene == state.applied.scene:
    old_version, old_settings = state.previous
    old_c = solve(old_settings)["C"]
    change_table = changed_orders(old_c, after)
    st.subheader(f"重排差异 · V{old_version:02} → {version}", anchor=False)
    st.caption(f"C 方案覆盖 {old_c.metrics.covered} → {after.metrics.covered} 单 · "
               f"可变费用 {(after.metrics.cost-old_c.metrics.cost)/100:+.2f} 元")
    if change_table.empty:
        st.info("订单归属和计划开工未发生变化。")
    else:
        st.dataframe(change_table, hide_index=True, width="stretch")

selected = plans[focused]
version_id = f"{version}-{focused}"
schedule, changes, costs = st.tabs(("路线甘特", "逐单明细", "班次费用"))
with schedule:
    st.caption(f"{version_id} · {PLAN_NAMES[focused]} · 到达、开工、返程使用同一份校验后的排程")
    st.plotly_chart(gantt(selected), width="stretch", config={"displayModeBar": False}, key="gantt")

worklist = order_detail(selected.scenario, selected.plan, version_id)
with changes:
    st.dataframe(worklist.drop(columns=["版本"]), hide_index=True, width="stretch")
with costs:
    bill = staff_bill(selected.scenario, selected.plan)
    st.dataframe(bill, hide_index=True, width="stretch")
    st.caption("费用由交通、全职加班与共享班次费组成。空档单独展示。")

if selected.plan.unassigned:
    missing = worklist.loc[worklist["人员"] == "", ["订单", "最迟开工", "未安排说明"]]
    st.subheader("未安排订单", anchor=False)
    st.dataframe(missing, hide_index=True, width="stretch")

with st.container(border=True):
    left, middle, right = st.columns([2, 1, 1], vertical_alignment="center")
    left.markdown(f"**工作单 · {version_id}**")
    left.caption(f"{len(worklist)} 笔订单 · 包含未安排行 · UTF-8 BOM")
    if middle.button("确认当前工作单", disabled=dirty, key="confirm", width="stretch"):
        state.confirmed = (state.version, focused)
    confirmed = state.confirmed == (state.version, focused) and not dirty
    right.download_button(
        "下载工作单 CSV", data=worklist.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"homecare_worklist_{version_id}.csv", mime="text/csv",
        disabled=not confirmed, key="download", width="stretch", on_click="ignore"
    )
    if confirmed:
        st.caption(f"{version_id} 已确认，可导出。")

with st.expander("算法对照 · WOA / PSO / WOA-PSO"):
    st.caption("在当前共享方案上，各跑三个相同的种子，每次评价 594 个候选。"
               "依次比覆盖、迟到与可变费用。")
    if st.button("运行算法对照", disabled=dirty, key="benchmark"):
        state.bench_settings = state.applied
    if state.get("bench_settings") == state.applied:
        with st.spinner("正在执行 9 次群体搜索…"):
            benchmark = algorithm_table(state.applied)
        st.dataframe(benchmark, hide_index=True, width="stretch")

st.caption("基于护理中心 H 竞赛案例构建。时间、路网、费率与结果均为仿真。")

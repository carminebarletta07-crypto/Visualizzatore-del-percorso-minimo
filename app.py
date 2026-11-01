
import streamlit as st
import streamlit.components.v1 as components
import time
import heapq
import math

from collections import deque
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROWS, COLS, CELL_SIZE = 20, 30, 30

path_grid = components.declare_component(
    "pathfinding_grid",
    path=str(Path(__file__).parent / "grid_component")
)

st.set_page_config(
    page_title="Pathfinding Visualizer",
    page_icon="🧭",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
        }

        div[data-testid=stButton] button {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INIZIALIZZAZIONE SESSION STATE
# ============================================================

def init():
    defaults = {
        "start": (1, 1),
        "end": (18, 28),
        "walls": set(),
        "visited": [],
        "path": [],
        "metrics": None,
        "animation_id": 0,
        "last_grid_event_id": -1
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init()


# ============================================================
# GESTIONE GRIGLIA
# ============================================================

def clear_results():
    st.session_state.visited = []
    st.session_state.path = []
    st.session_state.metrics = None


def clear_walls():
    st.session_state.walls = set()
    clear_results()


def reset_grid():
    st.session_state.start = (1, 1)
    st.session_state.end = (18, 28)
    st.session_state.walls = set()
    clear_results()


def apply_grid_event(event):
    """
    Applica un evento ricevuto dal componente JavaScript.
    Restituisce True se è stato ricevuto un nuovo evento valido.
    """

    if not isinstance(event, dict):
        return False

    event_id = event.get("event_id")

    if not isinstance(event_id, int):
        return False

    if event_id <= st.session_state.last_grid_event_id:
        return False

    try:
        walls = {
            tuple(map(int, point))
            for point in event.get("walls", [])
        }

        start = tuple(
            map(int, event.get("start", st.session_state.start))
        )

        end = tuple(
            map(int, event.get("end", st.session_state.end))
        )

        if len(start) != 2 or len(end) != 2:
            return False

        if not (
            0 <= start[0] < ROWS
            and 0 <= start[1] < COLS
        ):
            return False

        if not (
            0 <= end[0] < ROWS
            and 0 <= end[1] < COLS
        ):
            return False

        if start == end:
            return False

        walls = {
            (r, c)
            for r, c in walls
            if 0 <= r < ROWS and 0 <= c < COLS
        }

        walls.discard(start)
        walls.discard(end)

        # L'evento è stato elaborato
        st.session_state.last_grid_event_id = event_id

        changed = (
            walls != st.session_state.walls
            or start != st.session_state.start
            or end != st.session_state.end
        )

        if changed:
            st.session_state.walls = walls
            st.session_state.start = start
            st.session_state.end = end

            clear_results()

        return True

    except (TypeError, ValueError, IndexError):
        return False


# ============================================================
# ALGORITMI
# ============================================================

def neighbors(node, walls):
    r, c = node

    return [
        (nr, nc)
        for nr, nc in (
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        )
        if (
            0 <= nr < ROWS
            and 0 <= nc < COLS
            and (nr, nc) not in walls
        )
    ]


def reconstruct(came_from, node):
    result = []

    while node in came_from:
        result.append(node)
        node = came_from[node]

    return result[::-1]


def bfs(start, end, walls):
    queue = deque([start])
    came = {}
    seen = {start}
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        if node == end:
            return order, reconstruct(came, end)

        for nxt in neighbors(node, walls):
            if nxt not in seen:
                seen.add(nxt)
                came[nxt] = node
                queue.append(nxt)

    return order, []


def dijkstra(start, end, walls):
    dist = {start: 0}
    came = {}
    heap = [(0, start)]
    seen = set()
    order = []

    while heap:
        d, node = heapq.heappop(heap)

        if node in seen:
            continue

        seen.add(node)
        order.append(node)

        if node == end:
            return order, reconstruct(came, end)

        for nxt in neighbors(node, walls):
            nd = d + 1

            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                came[nxt] = node
                heapq.heappush(heap, (nd, nxt))

    return order, []


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, end, walls):
    g = {start: 0}
    came = {}

    heap = [
        (heuristic(start, end), 0, start)
    ]

    seen = set()
    order = []

    while heap:
        _, cost, node = heapq.heappop(heap)

        if node in seen:
            continue

        seen.add(node)
        order.append(node)

        if node == end:
            return order, reconstruct(came, end)

        for nxt in neighbors(node, walls):
            ng = cost + 1

            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                came[nxt] = node

                heapq.heappush(
                    heap,
                    (
                        ng + heuristic(nxt, end),
                        ng,
                        nxt
                    )
                )

    return order, []


def bellman_ford(start, end, walls):
    nodes = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) not in walls
    ]

    dist = {
        node: math.inf
        for node in nodes
    }

    came = {}
    dist[start] = 0

    edges = [
        (u, v, 1)
        for u in nodes
        for v in neighbors(u, walls)
    ]

    for _ in range(len(nodes) - 1):
        changed = False

        for u, v, w in edges:
            if (
                dist[u] != math.inf
                and dist[u] + w < dist[v]
            ):
                dist[v] = dist[u] + w
                came[v] = u
                changed = True

        if not changed:
            break

    order = [
        node
        for node in nodes
        if dist[node] != math.inf
    ]

    if dist[end] != math.inf:
        return order, reconstruct(came, end)

    return order, []


def run_algorithm(name, start, end, walls):
    algorithms = {
        "A*": astar,
        "Dijkstra": dijkstra,
        "BFS": bfs,
        "Bellman-Ford": bellman_ford
    }

    return algorithms[name](start, end, walls)


# ============================================================
# INTERFACCIA
# ============================================================

st.title("🧭 Pathfinding Visualizer")

st.caption(
    "Visualizzazione degli algoritmi di ricerca "
    "del percorso più breve su una griglia 30 × 20."
)

left, right = st.columns(
    [3.5, 1.2],
    gap="large"
)


# ============================================================
# CONTROLLI
# ============================================================

with right:

    st.subheader("Controlli")

    algorithm = st.selectbox(
        "Algoritmo",
        ["A*", "Dijkstra", "BFS", "Bellman-Ford"]
    )

    mode = st.selectbox(
        "Modalità di modifica",
        [
            "Disegna ostacoli",
            "Cancella ostacoli",
            "Imposta partenza",
            "Imposta destinazione"
        ]
    )

    speed = st.slider(
        "Velocità animazione",
        0.001,
        0.10,
        0.015,
        0.001
    )

    st.divider()

    st.subheader("Legenda")

    for item in [
        "🟩 Partenza",
        "🟥 Destinazione",
        "⬛ Ostacolo",
        "🟨 Nodo esplorato",
        "🟦 Percorso finale",
        "⬜ Cella libera"
    ]:
        st.markdown(item)

    st.divider()

    st.button(
        "🧹 Cancella ostacoli",
        on_click=clear_walls,
        use_container_width=True
    )

    st.button(
        "↩️ Ripristina griglia",
        on_click=reset_grid,
        use_container_width=True
    )

    st.divider()

    start_button = st.button(
        "▶ Avvia algoritmo",
        type="primary",
        use_container_width=True
    )


# ============================================================
# ESECUZIONE ALGORITMO
# ============================================================

if start_button:

    start = st.session_state.start
    end = st.session_state.end
    walls = set(st.session_state.walls)

    if start == end:
        st.error(
            "La partenza e la destinazione non possono coincidere."
        )

    else:
        t0 = time.perf_counter()

        visited, final_path = run_algorithm(
            algorithm,
            start,
            end,
            walls
        )

        elapsed = time.perf_counter() - t0

        st.session_state.visited = visited
        st.session_state.path = final_path

        st.session_state.metrics = {
            "algorithm": algorithm,
            "visited": len(visited),
            "path": len(final_path),
            "time": elapsed
        }

        st.session_state.animation_id += 1


# ============================================================
# GRIGLIA E SINCRONIZZAZIONE EVENTI
# ============================================================

with left:

    st.subheader("Griglia di ricerca")

    value = path_grid(
        rows=ROWS,
        cols=COLS,
        cell_size=CELL_SIZE,

        start=list(st.session_state.start),
        end=list(st.session_state.end),

        walls=[
            list(point)
            for point in sorted(st.session_state.walls)
        ],

        visited=[
            list(point)
            for point in st.session_state.visited
        ],

        path=[
            list(point)
            for point in st.session_state.path
        ],

        mode=mode,
        animation_id=st.session_state.animation_id,

        animation_nodes=[
            list(point)
            for point in st.session_state.visited
        ],

        animation_path=[
            list(point)
            for point in st.session_state.path
        ],

        speed_ms=max(1, int(speed * 1000)),

        key="pathfinding_grid"
    )

    # CORREZIONE SINCRONIZZAZIONE
    previous_event_id = st.session_state.last_grid_event_id

    new_event = apply_grid_event(value)

    if (
        new_event
        and st.session_state.last_grid_event_id > previous_event_id
    ):
        st.rerun()


# ============================================================
# RISULTATI
# ============================================================

if st.session_state.metrics is not None:

    m = st.session_state.metrics

    st.divider()
    st.subheader("Risultati")

    if m["path"]:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Nodi esplorati",
            m["visited"]
        )

        c2.metric(
            "Nodi del percorso",
            m["path"]
        )

        c3.metric(
            "Tempo di esecuzione",
            f'{m["time"] * 1000:.3f} ms'
        )

        st.success(
            f'Percorso trovato con {m["algorithm"]}.'
        )

    else:

        st.warning(
            f'Nessun percorso trovato con {m["algorithm"]}.'
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Nodi esplorati",
            m["visited"]
        )

        c2.metric(
            "Tempo di esecuzione",
            f'{m["time"] * 1000:.3f} ms'
        )


# ============================================================
# FIRMA
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        font-size:14px;
    ">
        Realizzata da Carmine Barletta
    </div>
    """,
    unsafe_allow_html=True
)
